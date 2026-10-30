// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-only PyO3 boundary for one materialized AISimulate replay execution.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::engine::{
    Backend, EngineConfig, IterationMs, MixedPassMs, PrefillTotals, RankWork,
    TimingEvidenceAccumulator, TimingEvidenceSource, TimingEvidenceSummary, TimingModel,
    TimingModelConfig, TimingOperationEvidence, TimingPhaseEvidence, ValidatedTimingPhase,
    VisionShape,
};
use crate::perfmodel::engine::runtime::{
    IterationDispatch, MixedPass, PerOpSlotValue, RankDetail, RankPricing,
};
use crate::perfmodel::engine::{Engine as PerfEngine, RuntimeConfig};
use crate::replay::{
    EncoderSpec, POWER_DATA_COVERAGE_THRESHOLD, ReplayArtifactKvEventVisibility, ReplayArtifacts,
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory,
    ReplayOperationPowerDiagnostics, ReplayPhasePowerDiagnostics, ReplayPowerDiagnostics,
    ReplayRoleConfig, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer, TracePowerStats,
    loadgen::{
        AgenticProfileOptions, AgenticSnapshotOptions, AgenticWarmupPolicy, ArrivalSpec, DelaySpec,
        DynamoRequestTrace, LengthSpec, SyntheticImages, SyntheticTraceSpec, Trace,
        ValidatedAgenticGraph, WekaImportOptions, WekaNestedTimestampBasis,
        WekaResolvedTimestampBasis, WorkloadDriver, load_agentic_mooncake,
        load_weka_agentic_graph_cached,
    },
};
use crate::{
    DeploymentLimits, EstimationMode, EstimatorConfig, ForwardPassFallbackPolicy,
    ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassRegressionWorkloadKind,
    ForwardPassWorkerType, UNKNOWN_SIZE,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExecutionPayload {
    Configured {
        spec: ReplaySpec,
        #[serde(default)]
        traffic: Option<Box<RuntimeTraffic>>,
        #[serde(default)]
        capture_performance_diagnostics: bool,
        #[serde(default)]
        determinism: ReplayDeterminism,
        #[serde(default)]
        tokenization: Option<TokenizationPayload>,
        #[serde(default)]
        detokenization: Option<DetokenizationPayload>,
    },
    Legacy(ReplaySpec),
}

/// Per-worker tokenization applied between placement and the engine.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizationPayload {
    #[serde(default = "default_tokenization_workers")]
    workers: usize,
    #[serde(default)]
    base_ms: f64,
    #[serde(default)]
    per_input_token_ms: f64,
}

fn default_tokenization_workers() -> usize {
    1
}

/// Detokenization on every token-emitting worker before the client.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct DetokenizationPayload {
    #[serde(default = "default_tokenization_workers")]
    workers: usize,
    #[serde(default)]
    base_ms: f64,
    #[serde(default)]
    per_output_ms: f64,
}

/// The request and response path around the engine: tokenization before it
/// and detokenization after it.
#[derive(Debug, Clone, Copy, Default)]
struct ServingPath {
    tokenization: Option<TokenizationPayload>,
    detokenization: Option<DetokenizationPayload>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTraffic {
    source_type: String,
    /// Configured deployment model used to time every agentic request. Source
    /// model labels remain provenance on the validated graph.
    #[serde(default)]
    execution_model: Option<String>,
    #[serde(default)]
    load_type: Option<String>,
    #[serde(default)]
    trace_path: Option<String>,
    #[serde(default)]
    trace_paths: Vec<String>,
    #[serde(default)]
    trace_format: Option<String>,
    #[serde(default)]
    trace_block_size: Option<usize>,
    #[serde(default)]
    weka_nested_timestamp_basis: Option<WekaNestedTimestampBasis>,
    /// Keep the loaded Weka graph for later runs in this process (sweeps).
    #[serde(default)]
    trace_graph_cache: bool,
    #[serde(default)]
    arrival_speedup_ratio: Option<f64>,
    #[serde(default)]
    replay_concurrency: Option<usize>,
    #[serde(default)]
    agentic_lanes: Option<usize>,
    #[serde(default)]
    agentic_snapshot: Option<AgenticSnapshotOptions>,
    #[serde(default)]
    agentic_warmup: bool,
    #[serde(default)]
    agentic_warmup_policy: AgenticWarmupPolicy,
    #[serde(default)]
    agentic_profile: Option<AgenticProfileOptions>,
    #[serde(default)]
    isl: Option<usize>,
    #[serde(default)]
    osl: Option<usize>,
    #[serde(default)]
    cached_prefix_tokens: Option<usize>,
    // Fixed per-request image workload resolved by the Python geometry layer.
    #[serde(default)]
    image_count: Option<usize>,
    #[serde(default)]
    image_visual_tokens: Option<usize>,
    #[serde(default)]
    image_sequences: Option<u32>,
    #[serde(default)]
    image_patch_tokens: Option<u32>,
    #[serde(default)]
    image_transformer_tokens: Option<u32>,
    #[serde(default)]
    image_output_tokens: Option<u32>,
    #[serde(default)]
    image_embedding_bytes: Option<u64>,
    #[serde(default)]
    image_identity_pool: Option<u64>,
    #[serde(default)]
    request_count: Option<usize>,
    #[serde(default)]
    turns_per_session: Option<usize>,
    #[serde(default)]
    shared_prefix_ratio: Option<f64>,
    #[serde(default)]
    num_prefix_groups: Option<usize>,
    #[serde(default)]
    inter_turn_delay_ms: Option<f64>,
    #[serde(default)]
    request_rate: Option<f64>,
    #[serde(default)]
    arrival_interval_ms: Option<f64>,
    #[serde(default)]
    arrival_seed: Option<u64>,
    #[serde(default)]
    concurrency: Option<usize>,
    // Fields consumed by the Python-side optimizer before execution.
    #[serde(default)]
    num_request_ratio: Option<f64>,
    #[serde(default)]
    kv_load_ratio: Option<serde_json::Value>,
    #[serde(default)]
    max_sim_time_ms: Option<f64>,
}

struct BuiltRuntimeInput {
    input: ReplayRuntimeInput,
    weka_nested_timestamp_basis: Option<WekaResolvedTimestampBasis>,
}

impl BuiltRuntimeInput {
    fn without_weka_basis(input: ReplayRuntimeInput) -> Self {
        Self {
            input,
            weka_nested_timestamp_basis: None,
        }
    }
}

const AGENTIC_MODEL_PROJECTION_POLICY: &str = "project_to_configured_target";

fn require_agentic_execution_model(traffic: &RuntimeTraffic) -> Result<&str> {
    traffic
        .execution_model
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .context("agentic execution requires a configured target model")
}

fn validate_public_agentic_engine(
    input: &ReplayRuntimeInput,
    spec: &ReplaySpec,
    engine: &ReplayEngineConfig,
) -> Result<()> {
    let ReplayRuntimeInput::Workload(driver) = input else {
        return Ok(());
    };
    if !driver.is_agentic() {
        return Ok(());
    }
    // A request parked in the pool is not yet a scheduler request, which the
    // agentic driver's in-flight accounting assumes.
    ensure!(
        spec.encoder.is_none(),
        "agentic replay does not combine with an encoder pool"
    );
    let roles = match &spec.topology {
        ReplayTopology::Aggregated { .. } => vec![aggregated_role(engine)],
        ReplayTopology::Disaggregated { .. } => vec![
            engine.role(crate::replay::WorkerStage::Prefill),
            engine.role(crate::replay::WorkerStage::Decode),
        ],
    };
    for role in &roles {
        ensure!(
            role.rank.g3_offload.is_none(),
            "agentic replay does not support G3 offload"
        );
        ensure!(
            matches!(role.rank.backend, Backend::Vllm | Backend::Sglang),
            "agentic replay supports only vLLM and SGLang backends"
        );
    }
    // HBM-only deployments retain their existing worker/DP support. Once any
    // active role enables G2, qualify the entire deployment, not just that rank.
    if roles
        .iter()
        .any(|role| role.rank.native_host_offload.is_some())
    {
        let single_worker_per_role = match &spec.topology {
            ReplayTopology::Aggregated { workers } => workers.initial_workers == 1,
            ReplayTopology::Disaggregated {
                prefill, decode, ..
            } => prefill.initial_workers == 1 && decode.initial_workers == 1,
        };
        ensure!(
            single_worker_per_role,
            "agentic host offload requires one aggregated worker or one prefill and one decode worker"
        );
        ensure!(
            spec.adapters.scaling.provider == "none",
            "agentic host offload requires static worker pools without a scaling policy"
        );
        for role in &roles {
            ensure!(
                role.rank.aic_nextn.is_none(),
                "agentic host offload requires speculative decoding disabled"
            );
            ensure!(
                role.rank.backend == Backend::Vllm,
                "agentic host offload requires backend=vllm on every role"
            );
            ensure!(
                role.dp_size == 1,
                "agentic host offload requires attention DP=1 on every role"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AicTimingConfig {
    #[serde(alias = "model_path")]
    model: String,
    backend: String,
    system: String,
    /// Sizes a saved identity records as `"unknown"` hold `UNKNOWN_SIZE`.
    #[serde(
        default = "one",
        alias = "tp_size",
        deserialize_with = "crate::perfmodel::fpm::identity_size::deserialize"
    )]
    tp: u32,
    #[serde(default)]
    backend_version: Option<String>,
    #[serde(
        default = "one",
        alias = "pp_size",
        deserialize_with = "crate::perfmodel::fpm::identity_size::deserialize"
    )]
    pp: u32,
    #[serde(
        default = "one",
        alias = "attention_dp_size",
        deserialize_with = "crate::perfmodel::fpm::identity_size::deserialize"
    )]
    attention_dp: u32,
    #[serde(default)]
    dcp: Option<u32>,
    #[serde(
        default,
        deserialize_with = "crate::perfmodel::fpm::identity_optional_size::deserialize"
    )]
    moe_tp_size: Option<u32>,
    #[serde(
        default,
        deserialize_with = "crate::perfmodel::fpm::identity_optional_size::deserialize"
    )]
    moe_ep_size: Option<u32>,
    /// Prefill context parallelism (SGLang attn-cp / vLLM PCP): folds into the
    /// attention width like attention_dp. `None` means 1.
    #[serde(default)]
    cp_size: Option<u32>,
    #[serde(default, alias = "gemm_quant_mode")]
    gemm_dtype: Option<String>,
    #[serde(default, alias = "moe_quant_mode")]
    moe_dtype: Option<String>,
    #[serde(default, alias = "fmha_quant_mode")]
    fmha_dtype: Option<String>,
    #[serde(default, alias = "fpm_fmha_quant_mode")]
    fpm_fmha_dtype: Option<String>,
    #[serde(default, alias = "kvcache_quant_mode")]
    kv_cache_dtype: Option<String>,
    #[serde(default, alias = "comm_quant_mode")]
    comm_dtype: Option<String>,
    #[serde(default)]
    nextn: u32,
    #[serde(default)]
    speculation: Option<crate::ForwardPassSpeculationConfig>,
    #[serde(default)]
    kv_block_size: Option<u32>,
    #[serde(default)]
    gpu_memory_utilization: Option<f64>,
    #[serde(default)]
    mem_fraction_static: Option<f64>,
    #[serde(default)]
    free_gpu_memory_fraction: Option<f64>,
    #[serde(default)]
    cuda_graph_reserved_bytes: u64,
    #[serde(default)]
    systems_path: Option<String>,
    #[serde(default)]
    forward_model: Option<String>,
    #[serde(default)]
    fpm_profile: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing)]
    fpm_interpolation: Option<crate::FpmInterpolationMethod>,
    #[serde(default)]
    fpm_parquet_path: Option<String>,
    #[serde(default)]
    decoder_replay: bool,
    /// Layout of the model's vision tower; required when the rank hosts it.
    #[serde(default)]
    encoder_parallel: Option<crate::EncoderParallel>,
    #[serde(default)]
    worker_type: Option<ForwardPassWorkerType>,
    #[serde(default)]
    estimation_mode: Option<EstimationMode>,
    #[serde(default)]
    fallback_policy: ForwardPassFallbackPolicy,
    #[serde(default)]
    estimator_config: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    database_mode: crate::DatabaseMode,
    #[serde(default)]
    transfer_policy: Option<Vec<String>>,
    #[serde(default)]
    systems_paths: Vec<PathBuf>,
    #[serde(default)]
    attention_backend: Option<String>,
    #[serde(default)]
    moe_kernel_source: Option<String>,
    #[serde(default)]
    moe_backend: Option<String>,
    #[serde(default)]
    enable_eplb: bool,
    #[serde(default)]
    wideep_num_slots: Option<u32>,
    #[serde(default, alias = "shared_layer")]
    enable_shared_layer: Option<bool>,
    #[serde(default)]
    strict_provenance: bool,
    #[serde(default)]
    deployment_limits: Option<DeploymentLimits>,
}

const fn one() -> u32 {
    1
}

impl AicTimingConfig {
    fn estimator_request(
        &self,
        worker_type: ForwardPassWorkerType,
    ) -> Result<ForwardPassPerfModelConfig> {
        ensure!(
            self.worker_type
                .is_none_or(|configured| configured == worker_type),
            "estimator worker_type does not match replay role"
        );
        let legacy_mode = match self.forward_model.as_deref() {
            Some("fpm") => Some(EstimationMode::FpmInterpolation),
            Some("op_level") => Some(EstimationMode::OpLevel),
            Some(other) => return Err(anyhow!("unknown legacy forward_model {other:?}")),
            None => None,
        };
        if let (Some(explicit), Some(legacy)) = (self.estimation_mode, legacy_mode) {
            ensure!(
                explicit == legacy,
                "forward_model conflicts with estimation_mode"
            );
        }
        let mode = self
            .estimation_mode
            .or(legacy_mode)
            .unwrap_or(if self.worker_type.is_some() {
                EstimationMode::Auto
            } else {
                EstimationMode::OpLevel
            });
        let roots = if self.systems_paths.is_empty() {
            self.systems_path.iter().map(PathBuf::from).collect()
        } else {
            self.systems_paths.clone()
        };
        if let Some(path) = &self.fpm_parquet_path {
            crate::config::validate_fpm_parquet_path(
                Some(std::path::Path::new(path)),
                mode == EstimationMode::FpmInterpolation,
            )?;
        }
        let mut estimator_input = serde_json::Value::Object(self.estimator_config.clone());
        EstimatorConfig::apply_context_defaults(&mut estimator_input, mode);
        let estimator_config = EstimatorConfig::migrate_legacy_inputs(
            Some(&serde_json::to_string(&estimator_input)?),
            None,
            self.fpm_interpolation.map(|method| method.as_str()),
            self.fpm_parquet_path.as_deref(),
        )?;
        Ok(ForwardPassPerfModelConfig {
            model: self.model.clone(),
            system: self.system.clone(),
            backend: serde_json::from_value(serde_json::Value::String(self.backend.clone()))?,
            backend_version: self.backend_version.clone(),
            worker_type,
            tp: self.tp,
            pp: self.pp,
            attention_dp: self.attention_dp,
            dcp: self.dcp,
            moe_tp_size: self.moe_tp_size,
            moe_ep_size: self.moe_ep_size,
            cp_size: self.cp_size,
            gemm_quant_mode: self.gemm_dtype.clone(),
            moe_quant_mode: self.moe_dtype.clone(),
            fmha_quant_mode: self.fmha_dtype.clone(),
            fpm_fmha_quant_mode: self.fpm_fmha_dtype.clone(),
            kvcache_quant_mode: self.kv_cache_dtype.clone(),
            comm_quant_mode: self.comm_dtype.clone(),
            nextn: self.nextn,
            speculation: self.speculation.clone(),
            kv_block_size: self.kv_block_size,
            decoder_replay: self.decoder_replay,
            estimation_mode: mode,
            fallback_policy: self.fallback_policy,
            estimator_config,
            fpm_profile: self.fpm_profile.clone(),
            database_mode: self.database_mode,
            transfer_policy: self.transfer_policy.clone(),
            systems_paths: roots,
            attention_backend: self.attention_backend.clone(),
            moe_backend: self.moe_backend.clone(),
            enable_eplb: self.enable_eplb,
            wideep_num_slots: self.wideep_num_slots,
            moe_kernel_source: self.moe_kernel_source.clone(),
            enable_shared_layer: self.enable_shared_layer,
            strict_provenance: self.strict_provenance,
            encoder_parallel: self.encoder_parallel,
            deployment_limits: self.deployment_limits,
        })
    }

    fn speculative_depth(&self) -> Result<u32> {
        if let Some(speculation) = &self.speculation {
            ensure!(self.nextn == 0, "speculation cannot be combined with nextn");
            match speculation.kind.as_str() {
                "none" => {}
                "ngram" => ensure!(
                    self.backend == "vllm",
                    "ngram speculation requires backend=vllm"
                ),
                _ => ensure!(
                    matches!(self.backend.as_str(), "vllm" | "sglang"),
                    "{} speculative replay requires backend=vllm or backend=sglang",
                    speculation.kind
                ),
            }
            return Ok(crate::py::speculation_replay_depth(speculation)?);
        }
        Ok(self.nextn)
    }

    fn resolved_backend_version(&self) -> &str {
        self.backend_version
            .as_deref()
            .unwrap_or(match self.backend.as_str() {
                "vllm" => "0.19.0",
                "sglang" => "0.5.10",
                "trtllm" => "1.3.0rc10",
                _ => "",
            })
    }

    fn resolved_memory_fraction(&self) -> Result<(&'static str, f64)> {
        for (name, value) in [
            ("gpu_memory_utilization", self.gpu_memory_utilization),
            ("mem_fraction_static", self.mem_fraction_static),
            ("free_gpu_memory_fraction", self.free_gpu_memory_fraction),
        ] {
            ensure!(
                value.is_none_or(|fraction| {
                    fraction.is_finite() && (0.0..=1.0).contains(&fraction)
                }),
                "{name} must be finite and between 0 and 1"
            );
        }
        match self.backend.as_str() {
            "vllm" => Ok(("of_total", self.gpu_memory_utilization.unwrap_or(0.9))),
            "sglang" => Ok(("of_total", self.mem_fraction_static.unwrap_or(0.88))),
            "trtllm" => Ok(("of_free", self.free_gpu_memory_fraction.unwrap_or(0.9))),
            _ => Err(anyhow!(
                "unsupported AIC backend {:?}; expected vllm, sglang, or trtllm",
                self.backend
            )),
        }
    }

    fn validate_parallel_shape(&self) -> Result<()> {
        ensure!(
            self.tp > 0
                && self.pp > 0
                && self.attention_dp > 0
                && self.moe_tp_size != Some(0)
                && self.moe_ep_size != Some(0)
                && self.cp_size != Some(0)
                && self.dcp != Some(0),
            "AIC timing parallel sizes tp, pp, attention_dp, moe_tp_size, \
             moe_ep_size, cp_size, and dcp must be positive"
        );
        ensure!(self.nextn <= 5, "AIC nextn must be in 0..=5");
        self.speculative_depth()?;
        ensure!(
            self.moe_tp_size.is_some() == self.moe_ep_size.is_some(),
            "AIC moe_tp_size and moe_ep_size must be configured together"
        );
        if let (Some(moe_tp), Some(moe_ep)) = (self.moe_tp_size, self.moe_ep_size) {
            // Prefill CP widens the attention side (mirrors ModelConfig's
            // `tp * attention_dp * cp == moe_tp * moe_ep`); decode CP reuses
            // ranks inside the attention group and is deliberately absent.
            let cp = u64::from(self.cp_size.unwrap_or(1));
            ensure!(
                self.fpm_profile.is_some()
                    || [self.tp, self.attention_dp, moe_tp, moe_ep].contains(&UNKNOWN_SIZE)
                    || u64::from(self.tp) * u64::from(self.attention_dp) * cp
                        == u64::from(moe_tp) * u64::from(moe_ep),
                "AIC topology requires tp * attention_dp * cp_size == moe_tp_size * moe_ep_size"
            );
        }
        Ok(())
    }
}

type PhaseEvidenceKey = (u32, u32, u32, u32, bool);

#[cfg(test)]
type PythonPerOpEntries = Vec<(String, f64, f64, String)>;

enum AicPhaseProvider {
    Native(Arc<PerfEngine>),
    /// A regression or learned estimator, which has no native engine and
    /// prices only whole iterations (see [`IterationQuery::Fitted`]).
    Fitted,
    /// Keep the original bridge as an independent test reference and probe seam.
    #[cfg(test)]
    Python,
    /// Inject native rows to test construction failures before cache publication.
    #[cfg(test)]
    NativeEntries(Vec<crate::perfmodel::engine::PerOpValue>),
}

/// What one recorded whole-model FPM estimate prices. The decode share of a
/// mixed pass differs from a separate decode pass at the same totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FpmEvidencePhase {
    Prefill,
    Decode,
    MixedDecode,
}

/// A recorded estimate's phase, three iteration totals and correction bits.
type FpmEvidenceKey = (FpmEvidencePhase, u32, u32, u32, u64);

/// The work kind of one rank's FPM record in a group iteration. A rank that
/// computes prefill and decode is reported with decode, as the decode share
/// of a mixed pass was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum IterationKind {
    Prefill,
    Decode,
    Mixed,
}

impl IterationKind {
    /// The kind of `query`'s work, or `None` for an idle rank.
    fn of(query: &RankQuery) -> Option<Self> {
        match (query[1] > 0, query[3] > 0) {
            (false, false) => None,
            (true, false) => Some(Self::Prefill),
            (false, true) => Some(Self::Decode),
            (true, true) => Some(Self::Mixed),
        }
    }
}

/// One rank's canonical query: `[num_prefill_requests, sum_prefill_tokens,
/// sum_prefill_kv_tokens, num_decode_requests, sum_decode_kv_tokens]`.
type RankQuery = [u32; 5];

/// A group iteration: every rank's query in DP order.
type IterationKey = Vec<RankQuery>;

/// A recorded rank estimate's kind, the rank's query and the bits of the
/// iteration's correction.
type FpmIterationKey = (IterationKind, RankQuery, u64);

/// How an attention-DP iteration is asked of the canonical model, decided
/// once at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IterationQuery {
    /// A native engine's convention (see [`IterationDispatch`]).
    Native(IterationDispatch),
    /// A regression or learned estimator, fitted to `ForwardPassMetrics`
    /// telemetry: each rank's scheduled totals as Dynamo's vLLM telemetry
    /// reports them, decode KV the requests' past KV, excluding each
    /// request's current token. `detail` adds the per-request lengths and
    /// length variances the estimator reads. Every iteration is priced here.
    Fitted { detail: bool },
}

/// Whole-iteration pricing through the canonical `ForwardPassMetrics` query,
/// decided once at construction.
struct IterationPricing {
    model: Arc<ForwardPassPerfModel>,
    query: IterationQuery,
    /// The mix-step pass of each operation an operation-graph rank reports,
    /// when every name belongs to one pass. Attributes a mixed rank's
    /// operations to prefill and decode evidence.
    mix_step_passes: Option<std::collections::HashMap<String, MixedPass>>,
    /// Priced iterations keyed by the complete ordered rank list. Absent while
    /// the model collects lookup coverage, which counts every invocation, and
    /// when the estimator reads per-request detail the key does not hold.
    cache: Option<quick_cache::sync::Cache<IterationKey, Arc<PricedIteration>>>,
    /// How a mixed operation-graph rank's operations split into prefill and
    /// decode evidence, per operation-name list the model reports.
    mixed_splits: Mutex<Vec<Arc<MixedSplit>>>,
    /// Each working operation-graph rank's evidence before correction, by
    /// its query and whether diagnostics are captured.
    rank_evidence: quick_cache::sync::Cache<(RankQuery, bool), Arc<RankEvidence>>,
}

/// One working rank's operation evidence before correction, each part with
/// whether it is prefill: one part for a pure rank, up to two for a mixed one.
type RankEvidence = Vec<(Arc<ValidatedTimingPhase>, bool)>;

/// The prefill and decode parts of a mixed operation-graph rank with the
/// operation names `names` (see [`AicTimingModel::rank_operation_evidence`]),
/// each over its own shared name list so that every mixed rank's part
/// accumulates in place.
struct MixedSplit {
    names: Arc<[String]>,
    /// Each operation's mix-step pass; `None` splits every operation by the
    /// rank's token share.
    passes: Option<Box<[MixedPass]>>,
    prefill: MixedPart,
    decode: MixedPart,
}

/// The operations of one part of a mixed rank: those with a positive share.
struct MixedPart {
    names: Arc<[String]>,
    indices: Box<[usize]>,
}

impl MixedSplit {
    /// Distinct name lists a replay remembers; an operation graph reports
    /// one for its mix step.
    const CAPACITY: usize = 8;

    fn new(
        names: &Arc<[String]>,
        passes: Option<&std::collections::HashMap<String, MixedPass>>,
    ) -> Self {
        let passes = passes.and_then(|passes| {
            names
                .iter()
                .map(|name| passes.get(name).copied())
                .collect::<Option<Box<[MixedPass]>>>()
        });
        // A mixed rank's token share is strictly between 0 and 1, so a part
        // keeps every operation except the other attention pass's.
        let part = |excluded: MixedPass| {
            let indices: Box<[usize]> = (0..names.len())
                .filter(|&index| {
                    passes
                        .as_ref()
                        .is_none_or(|passes| passes[index] != excluded)
                })
                .collect();
            let names = if indices.len() == names.len() {
                Arc::clone(names)
            } else {
                indices.iter().map(|&index| names[index].clone()).collect()
            };
            MixedPart { names, indices }
        };
        Self {
            prefill: part(MixedPass::DecodeAttention),
            decode: part(MixedPass::ContextAttention),
            names: Arc::clone(names),
            passes,
        }
    }

    /// The prefill fraction of operation `index` at `token_share`, as
    /// [`AicTimingModel::prefill_fractions`] assigns it.
    fn prefill_fraction(&self, index: usize, token_share: f64) -> f64 {
        match self.passes.as_ref().map(|passes| passes[index]) {
            Some(MixedPass::ContextAttention) => 1.0,
            Some(MixedPass::DecodeAttention) => 0.0,
            Some(MixedPass::SharedNonAttention) | None => token_share,
        }
    }
}

impl IterationPricing {
    /// Iteration pricing of a replay timing model over `native`, the engine
    /// of `model`, or `None` where replay keeps per-rank phase pricing: an
    /// engine without a common forward-pass query; bounded decoder replay,
    /// whose per-request extend geometry the aggregate rank query cannot
    /// carry; and an engine dispatch that does not match the timing model's
    /// phase convention, whole-model FPM decode totals or operation graphs.
    fn for_replay(
        model: &Arc<ForwardPassPerfModel>,
        native: &PerfEngine,
        use_fpm_decode_totals: bool,
        decoder_replay: bool,
    ) -> Result<Option<Self>> {
        let dispatch = native.iteration_dispatch();
        let enabled = match dispatch {
            _ if decoder_replay => false,
            IterationDispatch::Unsupported => false,
            IterationDispatch::WholeModel { .. } => use_fpm_decode_totals,
            IterationDispatch::Operations { .. } => !use_fpm_decode_totals,
        };
        if !enabled {
            return Ok(None);
        }
        Ok(Some(Self {
            model: Arc::clone(model),
            query: IterationQuery::Native(dispatch),
            mix_step_passes: native.mix_step_passes(),
            cache: model
                .fpm_query_coverage()?
                .is_none()
                .then(|| quick_cache::sync::Cache::new(256)),
            mixed_splits: Mutex::default(),
            rank_evidence: quick_cache::sync::Cache::new(1024),
        }))
    }

    /// The split of a mixed rank whose operations are named `names`.
    fn mixed_split(&self, names: &Arc<[String]>) -> Result<Arc<MixedSplit>> {
        let mut splits = self
            .mixed_splits
            .lock()
            .map_err(|_| anyhow!("mixed operation split cache was poisoned"))?;
        if let Some(split) = splits.iter().find(|split| Arc::ptr_eq(&split.names, names)) {
            return Ok(Arc::clone(split));
        }
        let split = Arc::new(MixedSplit::new(names, self.mix_step_passes.as_ref()));
        if splits.len() < MixedSplit::CAPACITY {
            splits.push(Arc::clone(&split));
        }
        Ok(split)
    }
}

/// One priced group iteration and the evidence each invocation records.
struct PricedIteration {
    total_ms: f64,
    /// The detailed whole-model FPM estimate, when diagnostics are captured.
    estimate: Option<crate::ForwardPassEstimate>,
    /// Operation evidence of every working rank, corrected, each with whether
    /// it is prefill. A mixed rank may contribute one record of each.
    evidence: Vec<(Arc<ValidatedTimingPhase>, bool)>,
}

struct AicTimingModel {
    engine: Py<PyAny>,
    diagnostic_model: Option<Arc<ForwardPassPerfModel>>,
    fpm_model: Option<Arc<ForwardPassPerfModel>>,
    /// The canonical model when it corrects native estimates, as a restored
    /// tuned snapshot does. Kept apart from the diagnostic handle so the
    /// correction applies whether or not evidence is captured.
    corrections: Option<Arc<ForwardPassPerfModel>>,
    decoder_replay: bool,
    phase_provider: AicPhaseProvider,
    use_fpm_decode_totals: bool,
    fpm_decode_kv_ceiling: Option<u32>,
    evidence: Mutex<TimingEvidenceAccumulator>,
    // Capture-only history. Keyed by phase, exact iteration totals and the
    // bits of the applied online correction, so repeated invocations
    // increment a count without cloning prior history.
    fpm_evidence: Mutex<std::collections::BTreeMap<FpmEvidenceKey, crate::FpmEstimateEvidence>>,
    /// Capture-only history of group iterations priced as one forward.
    fpm_iteration_evidence:
        Mutex<std::collections::BTreeMap<FpmIterationKey, crate::FpmEstimateEvidence>>,
    phase_cache: quick_cache::sync::Cache<PhaseEvidenceKey, Arc<ValidatedTimingPhase>>,
    /// Present when the rank hosts the vision tower.
    vision: Option<VisionTimer>,
    /// `None` keeps the per-rank phase pricing for every pass.
    iteration: Option<IterationPricing>,
}

/// The canonical estimator's compiled vision tower and the encoder evidence it
/// produced per image shape. Kept apart from the diagnostic handle, which
/// replay drops when no evidence capture was requested.
struct VisionTimer {
    model: Arc<ForwardPassPerfModel>,
    cache: quick_cache::sync::Cache<VisionShape, Arc<ValidatedTimingPhase>>,
}

impl AicTimingModel {
    /// `vision` marks a rank that hosts the vision tower: the canonical model
    /// must then have been asked to compile it (`encoder_parallel`).
    fn build(
        config: &mut AicTimingConfig,
        worker_type: ForwardPassWorkerType,
        vision: bool,
    ) -> Result<Self> {
        let request = config.estimator_request(worker_type)?;
        if matches!(
            request.estimation_mode,
            EstimationMode::FpmRegression | EstimationMode::FpmLearned
        ) {
            // Reject before a learned source is loaded.
            Self::require_fitted_role(config, request.estimation_mode, vision)?;
        }
        config.resolved_memory_fraction()?;
        if vision {
            ensure!(
                config.backend == "sglang"
                    && config.pp == 1
                    && config.attention_dp == 1
                    && config.cp_size.is_none_or(|cp| cp == 1)
                    && config.dcp.is_none_or(|dcp| dcp == 1)
                    && config.fpm_profile.is_none()
                    && config.nextn == 0
                    && config.speculation.is_none(),
                "AIC vision timing requires backend=sglang with pp=1, attention_dp=1, cp_size=1, dcp=1, no fpm_profile, and no speculative decoding"
            );
            ensure!(
                config.encoder_parallel.is_some(),
                "a rank hosting the vision encoder requires timing config encoder_parallel (tp or dp)"
            );
        }
        let model = Arc::new(
            ForwardPassPerfModel::best_available(request)
                .context("AIC timing provider could not construct the requested estimator")?,
        );
        let provenance = model
            .provenance()
            .context("canonical estimator omitted construction provenance")?;
        config.backend_version = provenance.config.backend_version.clone();
        config.speculation = provenance.config.speculation.clone();
        // Canonical construction validates and resolves cost identity first;
        // executor-specific width limits must not reclassify loader failures.
        config.validate_parallel_shape()?;
        config.systems_path = provenance
            .selected_systems_root
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned());
        config.systems_paths = provenance.config.systems_paths.clone();
        config.database_mode = provenance.config.database_mode;
        config.transfer_policy = provenance.config.transfer_policy.clone();
        config.estimation_mode = Some(provenance.selected_estimation_mode);
        config.worker_type = Some(worker_type);
        config.forward_model = None;
        config.fpm_parquet_path = None;
        config.fallback_policy = ForwardPassFallbackPolicy::Deny;
        config.estimator_config =
            serde_json::from_value(serde_json::to_value(&provenance.config.estimator_config)?)?;
        config.fpm_profile = provenance.config.fpm_profile.clone();
        config.fpm_interpolation = None;
        config.gemm_dtype = provenance.config.gemm_quant_mode.clone();
        config.moe_dtype = provenance.config.moe_quant_mode.clone();
        config.fmha_dtype = provenance.config.fmha_quant_mode.clone();
        config.kv_cache_dtype = provenance.config.kvcache_quant_mode.clone();
        config.comm_dtype = provenance.config.comm_quant_mode.clone();
        config.attention_backend = provenance.config.attention_backend.clone();
        let use_fpm_decode_totals =
            provenance.selected_estimation_mode == EstimationMode::FpmInterpolation;
        let Some(native) = model.native_engine() else {
            return Self::build_fitted(config, worker_type, vision, model);
        };
        if vision {
            ensure!(
                native.has_vision(),
                "AIC model {:?} compiled no vision tower for encoder_parallel {:?}",
                config.model,
                config.encoder_parallel
            );
        }
        let vision = vision.then(|| VisionTimer {
            model: Arc::clone(&model),
            cache: quick_cache::sync::Cache::new(64),
        });
        let phase_provider = AicPhaseProvider::Native(Arc::clone(&native));
        let iteration = IterationPricing::for_replay(
            &model,
            &native,
            use_fpm_decode_totals,
            config.decoder_replay,
        )?;
        let engine = Python::with_gil(|py| -> PyResult<_> {
            Ok(Py::new(py, crate::AicEngine::from_shared_engine(native))?.into_any())
        })?;
        let fpm_decode_kv_ceiling = if use_fpm_decode_totals {
            model.fpm_decode_kv_ceiling()?
        } else {
            None
        };
        Ok(Self {
            engine,
            fpm_model: use_fpm_decode_totals.then(|| Arc::clone(&model)),
            corrections: model
                .corrects_native_estimates()
                .then(|| Arc::clone(&model)),
            diagnostic_model: Some(model),
            decoder_replay: config.decoder_replay,
            phase_provider,
            use_fpm_decode_totals,
            fpm_decode_kv_ceiling,
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            fpm_evidence: Mutex::default(),
            fpm_iteration_evidence: Mutex::default(),
            phase_cache: quick_cache::sync::Cache::new(128),
            vision,
            iteration,
        })
    }

    /// A role a regression or learned estimator times. It prices every pass
    /// through [`TimingModel::predict_iteration_ms`] and never a phase, so
    /// work the iteration query cannot describe is rejected here: a vision
    /// tower, speculative verification widths and bounded decoder replay.
    fn require_fitted_role(
        config: &AicTimingConfig,
        mode: EstimationMode,
        vision: bool,
    ) -> Result<()> {
        let mode = wire_name(mode);
        ensure!(
            !vision,
            "{mode} cannot time a rank hosting the vision encoder; it prices only language-model iterations"
        );
        ensure!(
            config.nextn == 0
                && config
                    .speculation
                    .as_ref()
                    .is_none_or(|speculation| speculation.kind == "none"),
            "{mode} replay does not support speculative decoding (nextn or speculation); \
             the iteration query has no verification width"
        );
        ensure!(
            !config.decoder_replay,
            "{mode} replay does not support bounded decoder replay"
        );
        Ok(())
    }

    /// The timing model of a regression or learned estimator: latency-only,
    /// priced once per group pass from the iteration query.
    fn build_fitted(
        config: &AicTimingConfig,
        worker_type: ForwardPassWorkerType,
        vision: bool,
        model: Arc<ForwardPassPerfModel>,
    ) -> Result<Self> {
        let provenance = model
            .provenance()
            .context("canonical estimator omitted construction provenance")?;
        let mode = provenance.selected_estimation_mode;
        ensure!(
            matches!(
                mode,
                EstimationMode::FpmRegression | EstimationMode::FpmLearned
            ),
            "canonical {mode:?} estimator has no native engine"
        );
        if provenance.requested_estimation_mode != mode {
            let failures = provenance.selection_failures.join("; ");
            return Err(anyhow!(
                "no native estimator is available for replay ({failures}); replay does not fall back to an \
                 untrained fpm_regression model"
            ));
        }
        Self::require_fitted_role(config, mode, vision)?;
        require_iteration_stores(&model, worker_type, config.attention_dp)?;
        let detail = model.reads_request_detail();
        let iteration = IterationPricing {
            model: Arc::clone(&model),
            query: IterationQuery::Fitted { detail },
            mix_step_passes: None,
            cache: (!detail).then(|| quick_cache::sync::Cache::new(256)),
            // Fitted estimators report no operation evidence.
            mixed_splits: Mutex::default(),
            rank_evidence: quick_cache::sync::Cache::new(1),
        };
        Ok(Self {
            engine: Python::with_gil(|py| py.None()),
            diagnostic_model: None,
            fpm_model: None,
            corrections: None,
            decoder_replay: false,
            phase_provider: AicPhaseProvider::Fitted,
            use_fpm_decode_totals: false,
            fpm_decode_kv_ceiling: None,
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            fpm_evidence: Mutex::default(),
            fpm_iteration_evidence: Mutex::default(),
            phase_cache: quick_cache::sync::Cache::new(1),
            vision: None,
            iteration: Some(iteration),
        })
    }

    fn is_fitted(&self) -> bool {
        matches!(self.phase_provider, AicPhaseProvider::Fitted)
    }

    /// Phase pricing is never a fallback for an estimator fitted to whole
    /// iterations, which would otherwise price work it was not fitted on.
    fn require_phase_pricing(&self) -> Result<()> {
        ensure!(
            !self.is_fitted(),
            "a regression or learned estimator prices only whole iterations; replay asked it for a separate phase"
        );
        Ok(())
    }

    /// Operation evidence of one encoder call over `shape.count` equal images,
    /// priced by the canonical model's compiled vision tower.
    fn vision_phase(&self, shape: VisionShape) -> Result<Arc<ValidatedTimingPhase>> {
        let timer = self
            .vision
            .as_ref()
            .context("this rank does not host the vision encoder")?;
        if let Some(phase) = timer.cache.get(&shape) {
            return Ok(phase);
        }
        let entries = timer
            .model
            .vision_operations(&[crate::EncoderImageShape {
                sequences: shape.encoder.sequences,
                patch_tokens: shape.encoder.patch_tokens,
                transformer_tokens: shape.encoder.transformer_tokens,
                output_tokens: shape.encoder.output_tokens,
                images: shape.count,
            }])
            .map_err(|error| anyhow!("AIC vision evaluation failed: {error}"))?;
        ensure!(
            !entries.is_empty(),
            "AIC vision encoder returned no operation evidence for {} image(s)",
            shape.count
        );
        let phase = Arc::new(phase_evidence_from_native_entries(entries)?);
        timer.cache.insert(shape, Arc::clone(&phase));
        Ok(phase)
    }

    fn predict_phase_evidence(
        &self,
        batch_size: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        mode: &str,
    ) -> Result<Arc<ValidatedTimingPhase>> {
        let prefill = mode == "static_ctx";
        if batch_size == 0 || (prefill && isl <= prefix) {
            static EMPTY: std::sync::OnceLock<Arc<ValidatedTimingPhase>> =
                std::sync::OnceLock::new();
            return Ok(Arc::clone(EMPTY.get_or_init(Default::default)));
        }
        let key = (batch_size, isl, osl, prefix, prefill);
        if let Some(phase) = self.phase_cache.get(&key) {
            return Ok(phase);
        }
        if let Some(model) = &self.diagnostic_model {
            let entries = model.static_phase_diagnostics(batch_size, isl, prefix, prefill)?;
            ensure!(
                !entries.is_empty(),
                "AIC {mode} returned empty operation diagnostics for nonzero work"
            );
            let operations = entries
                .into_iter()
                .map(|entry| {
                    let mut operation = TimingOperationEvidence::new(
                        entry.name,
                        entry.latency_ms,
                        Some(entry.energy_wms),
                        TimingEvidenceSource::from_provider(entry.source),
                    )?;
                    operation.details = Some(entry.details);
                    Ok(operation)
                })
                .collect::<Result<Vec<_>>>()?;
            // Diagnostics are name-folded in first-encounter order.
            let phase = Arc::new(ValidatedTimingPhase::from_name_folded_operations(
                operations,
            )?);
            self.phase_cache.insert(key, Arc::clone(&phase));
            return Ok(phase);
        }
        let phase = match &self.phase_provider {
            AicPhaseProvider::Fitted => {
                return Err(anyhow!(
                    "a regression or learned estimator has no {mode} operation evidence"
                ));
            }
            #[cfg(test)]
            AicPhaseProvider::NativeEntries(entries) => {
                ensure!(
                    !entries.is_empty(),
                    "AIC {mode} returned empty operation evidence for nonzero work"
                );
                phase_evidence_from_native_entries(entries.clone())?
            }
            AicPhaseProvider::Native(engine) => {
                let runtime = RuntimeConfig {
                    batch_size,
                    beam_width: 1,
                    isl,
                    osl,
                    prefix,
                    seq_imbalance_correction_scale: 1.0,
                    gen_seq_imbalance_correction_scale: 1.0,
                };
                let (names, values) = crate::py::parse_mode(mode)
                    .and_then(|mode| {
                        engine.reset_provenance();
                        engine
                            .run_static_slot_values(
                                &runtime,
                                mode,
                                crate::perfmodel::engine::DEFAULT_STATIC_STRIDE,
                            )
                            .map_err(crate::py::aic_to_py)
                    })
                    .map_err(|error| anyhow!("AIC {mode} evidence prediction failed: {error}"))?;
                ensure!(
                    !values.is_empty(),
                    "AIC {mode} returned empty operation evidence for nonzero work"
                );
                phase_evidence_from_native_slots(names, values)?
            }
            #[cfg(test)]
            AicPhaseProvider::Python => {
                let (context, generation) = self
                    .phase_evidence_through_python(batch_size, isl, osl, prefix, mode)
                    .map_err(|error| anyhow!("AIC {mode} evidence prediction failed: {error}"))?;
                let entries = if prefill { context } else { generation };
                ensure!(
                    !entries.is_empty(),
                    "AIC {mode} returned empty operation evidence for nonzero work"
                );
                phase_evidence_from_entries(entries)?
            }
        };
        let phase = Arc::new(phase);
        self.phase_cache.insert(key, Arc::clone(&phase));
        Ok(phase)
    }

    #[cfg(test)]
    fn phase_evidence_through_python(
        &self,
        batch_size: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        mode: &str,
    ) -> PyResult<(PythonPerOpEntries, PythonPerOpEntries)> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("batch_size", batch_size)?;
            kwargs.set_item("beam_width", 1)?;
            kwargs.set_item("isl", isl)?;
            kwargs.set_item("osl", osl)?;
            kwargs.set_item("prefix", prefix)?;
            kwargs.set_item("seq_imbalance_correction_scale", 1.0)?;
            kwargs.set_item("gen_seq_imbalance_correction_scale", 1.0)?;
            kwargs.set_item("mode", mode)?;
            kwargs.set_item("stride", crate::perfmodel::engine::DEFAULT_STATIC_STRIDE)?;
            self.engine
                .bind(py)
                .call_method("run_static_per_op", (), Some(&kwargs))?
                .extract::<(PythonPerOpEntries, PythonPerOpEntries)>()
        })
    }

    fn record_evidence(&self, phase: &ValidatedTimingPhase, prefill: bool) -> Result<()> {
        let mut evidence = self
            .evidence
            .lock()
            .map_err(|_| anyhow!("AIC timing evidence accumulator was poisoned"))?;
        evidence.record(phase, prefill)
    }

    fn record_fpm_estimate(
        &self,
        (phase, first, second, third): (FpmEvidencePhase, u32, u32, u32),
        estimate: crate::ForwardPassEstimate,
    ) -> Result<f64> {
        let latency_ms = estimate
            .latency_ms
            .context("native FPM returned no estimate")?;
        // Empty work runs no lookup. A mixed decode share can be zero after
        // real decode and baseline lookups, and those remain evidence.
        let looked_up_baseline = estimate
            .ranks
            .iter()
            .any(|rank| rank.decode_baseline_ms.is_some());
        if latency_ms > 0.0 || looked_up_baseline {
            let mut evidence = self
                .fpm_evidence
                .lock()
                .map_err(|_| anyhow!("FPM evidence accumulator was poisoned"))?;
            // A tuned model corrects the same prefill totals differently alone
            // and inside a mixed pass, so the applied correction is part of
            // the record's identity.
            let correction = estimate.correction_factor.map_or(0, f64::to_bits);
            let key = (phase, first, second, third, correction);
            let entry = evidence.entry(key).or_insert(crate::FpmEstimateEvidence {
                estimate,
                count: 0,
                latency_scale: 1.0,
            });
            entry.count = entry
                .count
                .checked_add(1)
                .context("FPM evidence invocation count overflow")?;
        }
        Ok(latency_ms)
    }

    /// The canonical model's online correction for the iteration `scheduled`
    /// describes. Replay prices passes from the native engine, so a tuned
    /// model's correction, such as a restored snapshot's, is applied here
    /// exactly as its own estimates apply it, with or without captured
    /// evidence: the pure prefill or decode correction for a pass with one
    /// kind of work and the mixed correction for a pass with both.
    fn phase_correction(
        &self,
        scheduled: impl FnOnce() -> Result<crate::ScheduledRequestMetrics>,
    ) -> Result<f64> {
        let Some(model) = &self.corrections else {
            return Ok(1.0);
        };
        let metrics = crate::ForwardPassMetrics {
            scheduled_requests: scheduled()?,
            ..Default::default()
        };
        Ok(model.native_correction_factor(&[metrics])?)
    }

    fn prefill_correction(
        &self,
        batch_size: usize,
        new_tokens: usize,
        kv_read_tokens: usize,
    ) -> Result<f64> {
        self.phase_correction(|| {
            Ok(crate::ScheduledRequestMetrics {
                num_prefill_requests: checked_u32(batch_size, "prefill batch size")?,
                sum_prefill_tokens: checked_u32(new_tokens, "total prefill tokens")?,
                sum_prefill_kv_tokens: checked_u32(kv_read_tokens, "total prefill KV tokens")?,
                ..Default::default()
            })
        })
    }

    fn decode_correction(
        &self,
        batch_size: usize,
        total_past_kv_tokens: impl FnOnce() -> Result<usize>,
    ) -> Result<f64> {
        self.phase_correction(|| {
            Ok(crate::ScheduledRequestMetrics {
                num_decode_requests: checked_u32(batch_size, "decode batch size")?,
                sum_decode_kv_tokens: checked_u32(total_past_kv_tokens()?, "total past KV tokens")?,
                ..Default::default()
            })
        })
    }

    /// The online correction of one forward that computes `prefill` together
    /// with `batch_size` decode requests reading `total_past_kv_tokens`: the
    /// factor of that mixed iteration, not of either phase alone.
    fn mixed_correction(
        &self,
        prefill: PrefillTotals,
        batch_size: usize,
        total_past_kv_tokens: usize,
    ) -> Result<f64> {
        self.phase_correction(|| {
            Ok(crate::ScheduledRequestMetrics {
                num_prefill_requests: checked_u32(prefill.batch_size, "prefill batch size")?,
                sum_prefill_tokens: checked_u32(prefill.new_tokens, "total prefill tokens")?,
                sum_prefill_kv_tokens: checked_u32(
                    prefill.prefix_tokens,
                    "total prefill KV tokens",
                )?,
                num_decode_requests: checked_u32(batch_size, "decode batch size")?,
                sum_decode_kv_tokens: checked_u32(total_past_kv_tokens, "total past KV tokens")?,
                ..Default::default()
            })
        })
    }

    /// Prefill at per-request mean shapes with `correction` applied.
    fn native_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
        correction: f64,
    ) -> Result<f64> {
        let batch_size = checked_u32(batch_size, "prefill batch size")?;
        let mean_isl = checked_u32(mean_isl, "mean input length")?;
        let mean_prefix = checked_u32(mean_prefix, "mean prefix length")?;
        if !self.use_fpm_decode_totals {
            let evidence = Self::corrected_phase(
                self.predict_phase_evidence(batch_size, mean_isl, 1, mean_prefix, "static_ctx")?,
                correction,
            )?;
            let latency_ms = evidence.latency_ms();
            self.record_evidence(&evidence, true)?;
            return Ok(latency_ms);
        }
        if let Some(model) = &self.diagnostic_model {
            return self.record_fpm_estimate(
                (FpmEvidencePhase::Prefill, batch_size, mean_isl, mean_prefix),
                corrected_estimate(
                    model.static_prefill_detailed(batch_size, mean_isl, mean_prefix)?,
                    correction,
                ),
            );
        }
        if let Some(model) = &self.fpm_model {
            return model
                .predict_prefill_latency(batch_size, mean_isl, mean_prefix)
                .map(|native| native * correction)
                .context("AIC prefill prediction failed");
        }
        Python::with_gil(|py| {
            self.engine
                .bind(py)
                .call_method1(
                    "predict_prefill_latency",
                    (batch_size, mean_isl, mean_prefix),
                )?
                .extract::<f64>()
        })
        .map(|native| native * correction)
        .map_err(|error| anyhow!("AIC prefill prediction failed: {error}"))
    }

    /// Operation evidence of a phase stretched by `correction`.
    fn corrected_phase(
        evidence: Arc<ValidatedTimingPhase>,
        correction: f64,
    ) -> Result<Arc<ValidatedTimingPhase>> {
        if correction == 1.0 {
            return Ok(evidence);
        }
        Ok(Arc::new(evidence.scaled(correction)?))
    }

    /// One rank's canonical query in the selected estimator's convention, or
    /// `None` when that query cannot represent the rank's work.
    ///
    /// Prefill passes its exact totals. Whole-model FPM reads decode past KV,
    /// excluding each request's current token and clamped like the phase
    /// path. Operation graphs read packed decode counters: `B * v` requests
    /// at `s` tokens each, `s` the integer mean context including the current
    /// token, as the static decode step prices `B` requests widened by `v`.
    fn rank_query(dispatch: IterationDispatch, rank: &RankWork) -> Result<Option<RankQuery>> {
        let mut query = [0; 5];
        if let Some(prefill) = &rank.prefill
            && prefill.totals.batch_size > 0
            && prefill.totals.new_tokens > 0
        {
            let totals = prefill.totals;
            let (Ok(batch), Ok(new), Ok(prefix)) = (
                u32::try_from(totals.batch_size),
                u32::try_from(totals.new_tokens),
                u32::try_from(totals.prefix_tokens),
            ) else {
                return Ok(None);
            };
            query[..3].copy_from_slice(&[batch, new, prefix]);
        }
        if let Some(decode) = &rank.decode
            && decode.batch_size > 0
        {
            let (requests, kv_tokens) = match dispatch {
                IterationDispatch::WholeModel { .. } => (
                    decode.batch_size,
                    total_past_kv_tokens(
                        decode.batch_size,
                        decode.active_kv_tokens,
                        decode.kv_bound_tokens,
                    )?,
                ),
                IterationDispatch::Operations { verify_width, .. } => {
                    if usize::try_from(verify_width).ok() != Some(decode.verify_width) {
                        // The static decode step widens by the graph's own
                        // depth; a scheduler at another depth keeps it.
                        return Ok(None);
                    }
                    ensure!(
                        decode.mean_context_length > 0,
                        "mean decode context must include the current input token"
                    );
                    let Some(requests) = decode.batch_size.checked_mul(decode.verify_width) else {
                        return Ok(None);
                    };
                    let Some(kv_tokens) = requests.checked_mul(decode.mean_context_length) else {
                        return Ok(None);
                    };
                    (requests, kv_tokens)
                }
                IterationDispatch::Unsupported => return Ok(None),
            };
            let (Ok(requests), Ok(kv_tokens)) = (u32::try_from(requests), u32::try_from(kv_tokens))
            else {
                return Ok(None);
            };
            query[3] = requests;
            query[4] = kv_tokens;
        }
        Ok(Some(query))
    }

    fn rank_metrics(query: &RankQuery) -> crate::ForwardPassMetrics {
        let [
            num_prefill_requests,
            sum_prefill_tokens,
            sum_prefill_kv_tokens,
            num_decode_requests,
            sum_decode_kv_tokens,
        ] = *query;
        crate::ForwardPassMetrics {
            scheduled_requests: crate::ScheduledRequestMetrics {
                num_prefill_requests,
                sum_prefill_tokens,
                sum_prefill_kv_tokens,
                num_decode_requests,
                sum_decode_kv_tokens,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Price one iteration with the canonical model: the plain estimate of
    /// whole-model FPM, its detailed estimate while diagnostics are captured,
    /// and for operation graphs the estimate with each rank's operations,
    /// which are the power evidence, as values or, while diagnostics are
    /// captured, diagnostics. Corrections are the model's own: the heaviest
    /// rank's workload correction applied to the rank maximum.
    fn price_iteration(
        &self,
        pricing: &IterationPricing,
        queries: &[RankQuery],
        ranks: &[RankWork],
        capture: bool,
    ) -> Result<PricedIteration> {
        let dispatch = match pricing.query {
            IterationQuery::Native(dispatch) => dispatch,
            IterationQuery::Fitted { detail } => {
                return Self::price_fitted_iteration(&pricing.model, queries, ranks, detail);
            }
        };
        let model = &pricing.model;
        let metrics: Vec<_> = queries.iter().map(Self::rank_metrics).collect();
        let whole_model = matches!(dispatch, IterationDispatch::WholeModel { .. });
        if whole_model && !capture {
            let total_ms = model
                .estimate_forward_pass_time_ms(&metrics)
                .context("AIC iteration prediction failed")?
                .context("canonical model returned no iteration estimate")?;
            return Ok(PricedIteration {
                total_ms,
                estimate: None,
                evidence: Vec::new(),
            });
        }
        let detail = if capture {
            RankDetail::Diagnostics
        } else {
            RankDetail::Values
        };
        if !whole_model
            && let Some(iteration) = model
                .estimate_native_iteration(&metrics, detail)
                .context("AIC iteration prediction failed")?
        {
            return Ok(PricedIteration {
                total_ms: iteration.latency_ms,
                evidence: self.native_rank_evidence(pricing, &iteration, queries, capture)?,
                estimate: None,
            });
        }
        let estimate = model
            .estimate_forward_pass_detailed(&metrics)
            .context("AIC iteration prediction failed")?;
        let total_ms = estimate
            .latency_ms
            .context("canonical model returned no iteration estimate")?;
        if whole_model {
            return Ok(PricedIteration {
                total_ms,
                estimate: Some(estimate),
                evidence: Vec::new(),
            });
        }
        Ok(PricedIteration {
            total_ms,
            evidence: self.rank_operation_evidence(pricing, &estimate, queries)?,
            estimate: None,
        })
    }

    /// Price one iteration with a regression or learned estimator: one plain
    /// estimate of every rank. The role's stores were checked at
    /// construction, so a missing estimate is an error, never another
    /// estimator. Latency-only: no evidence is recorded.
    fn price_fitted_iteration(
        model: &ForwardPassPerfModel,
        queries: &[RankQuery],
        ranks: &[RankWork],
        detail: bool,
    ) -> Result<PricedIteration> {
        let metrics = Self::fitted_metrics(queries, ranks, detail)?;
        let total_ms = model
            .estimate_forward_pass_time_ms(&metrics)
            .context("AIC iteration prediction failed")?
            .with_context(|| {
                format!("the regression or learned model has no iteration estimate for {metrics:?}")
            })?;
        Ok(PricedIteration {
            total_ms,
            estimate: None,
            evidence: Vec::new(),
        })
    }

    /// The group's metrics for a regression or learned estimator, one per
    /// rank in DP-rank order, idle ranks included. Each carries its rank's
    /// position as `dp_rank`, as an attention-DP group's telemetry does; a
    /// learned model's mixed composition validates complete rank groups.
    fn fitted_metrics(
        queries: &[RankQuery],
        ranks: &[RankWork],
        detail: bool,
    ) -> Result<Vec<crate::ForwardPassMetrics>> {
        queries
            .iter()
            .zip(ranks)
            .enumerate()
            .map(|(index, (query, rank))| {
                let mut metrics = if detail {
                    Self::detailed_rank_metrics(query, rank)?
                } else {
                    Self::rank_metrics(query)
                };
                metrics.dp_rank =
                    u32::try_from(index).context("attention-DP rank overflows u32")?;
                Ok(metrics)
            })
            .collect()
    }

    /// One rank's query for a regression or learned estimator: its scheduled
    /// totals as the `ForwardPassMetrics` telemetry such models are fitted to
    /// reports them. Dynamo's vLLM telemetry sums each decode request's
    /// computed tokens, so decode KV is the requests' past KV, their context
    /// lengths minus one current token each, as whole-model FPM reads it but
    /// without the scheduler's KV clamp. No other query prices the pass, so
    /// totals beyond the 32-bit counters are an error.
    fn fitted_rank_query(rank: &RankWork) -> Result<RankQuery> {
        let mut query = [0; 5];
        if let Some(prefill) = &rank.prefill
            && prefill.totals.batch_size > 0
            && prefill.totals.new_tokens > 0
        {
            let totals = prefill.totals;
            query[0] = checked_u32(totals.batch_size, "prefill batch size")?;
            query[1] = checked_u32(totals.new_tokens, "total prefill tokens")?;
            query[2] = checked_u32(totals.prefix_tokens, "total prefill KV tokens")?;
        }
        if let Some(decode) = &rank.decode
            && decode.batch_size > 0
        {
            ensure!(
                decode.verify_width == 1,
                "a regression or learned estimator cannot price speculative verification"
            );
            let past_kv_tokens = decode
                .active_kv_tokens
                .checked_sub(decode.batch_size)
                .context("active decode tokens must include one current token per request")?;
            query[3] = checked_u32(decode.batch_size, "decode batch size")?;
            query[4] = checked_u32(past_kv_tokens, "total decode past KV tokens")?;
        }
        Ok(query)
    }

    /// `query`'s metrics with the per-request detail of `rank`'s work in it:
    /// each prefill request's new tokens and prefix, and each decode
    /// request's one token and past KV (its context excluding the current
    /// token), as `extend_lengths` / `past_kv_lengths`; and the population
    /// variances of the prefill requests' full prompt lengths, as Dynamo's
    /// vLLM telemetry and the canonical field report them whatever chunk of
    /// the prompt a pass computes, and of the decode requests' past KV, the
    /// totals' decode KV convention.
    fn detailed_rank_metrics(
        query: &RankQuery,
        rank: &RankWork,
    ) -> Result<crate::ForwardPassMetrics> {
        fn variance(values: impl Iterator<Item = usize> + Clone) -> f64 {
            let count = values.clone().count() as f64;
            let mean = values.clone().map(|value| value as f64).sum::<f64>() / count;
            values
                .map(|value| (value as f64 - mean).powi(2))
                .sum::<f64>()
                / count
        }
        let mut metrics = Self::rank_metrics(query);
        let scheduled = &mut metrics.scheduled_requests;
        let (mut extend, mut past) = (Vec::new(), Vec::new());
        if query[1] > 0 {
            let prefill = rank
                .prefill
                .as_ref()
                .context("prefill query without prefill work")?;
            ensure!(
                prefill.requests.len() == prefill.totals.batch_size,
                "rank {} reported {} prefill requests for a batch of {}",
                rank.dp_rank,
                prefill.requests.len(),
                prefill.totals.batch_size
            );
            for &(new_tokens, prefix_tokens, _) in &prefill.requests {
                extend.push(new_tokens as u64);
                past.push(prefix_tokens as u64);
            }
            scheduled.var_prefill_length = variance(
                prefill
                    .requests
                    .iter()
                    .map(|&(_, _, prompt_len)| prompt_len),
            );
        }
        if query[3] > 0 {
            let decode = rank
                .decode
                .as_ref()
                .context("decode query without decode work")?;
            ensure!(
                decode.lengths.len() == decode.batch_size
                    && decode.lengths.iter().all(|length| *length > 0),
                "rank {} reported {} decode context lengths for a batch of {}, each including its current token",
                rank.dp_rank,
                decode.lengths.len(),
                decode.batch_size
            );
            for &length in &decode.lengths {
                extend.push(1);
                past.push(length as u64 - 1);
            }
            scheduled.var_decode_kv_tokens =
                variance(decode.lengths.iter().map(|length| length - 1));
        }
        if !extend.is_empty() {
            scheduled.extend_lengths = Some(extend);
            scheduled.past_kv_lengths = Some(past);
        }
        Ok(metrics)
    }

    /// Power evidence of an operation-graph iteration: every working rank's
    /// operations from the detailed `estimate`, stretched by the iteration's
    /// online correction as a phase's evidence is, so each rank's energy
    /// counts as it did when each rank priced its own phases.
    ///
    /// A rank that computes one kind of work is recorded with that phase. A
    /// rank that computes prefill and decode in one mix step is attributed by
    /// pass: its context-attention operations are prefill, its decode-attention
    /// operations decode, and each operation of the shared non-attention pass
    /// splits by the rank's token share, `ctx_tokens / (ctx_tokens +
    /// gen_tokens)` prefill. When an operation name may belong to more than
    /// one pass (see [`PerfEngine::mix_step_passes`]), the whole rank splits
    /// by that token share. A split scales an operation's latency, energy and
    /// SOL alike, keeping its power and the rank's totals.
    fn rank_operation_evidence(
        &self,
        pricing: &IterationPricing,
        estimate: &crate::ForwardPassEstimate,
        queries: &[RankQuery],
    ) -> Result<Vec<(Arc<ValidatedTimingPhase>, bool)>> {
        ensure!(
            estimate.ranks.len() == queries.len(),
            "canonical model did not report every rank of an operation-graph iteration"
        );
        let correction = estimate.correction_factor.unwrap_or(1.0);
        let mut evidence = Vec::with_capacity(queries.len());
        for (index, (rank, query)) in estimate.ranks.iter().zip(queries).enumerate() {
            let Some(kind) = IterationKind::of(query) else {
                continue;
            };
            ensure!(
                !rank.operations.is_empty(),
                "canonical model reported no operations for rank {index}'s work"
            );
            let prefill_fractions = Self::prefill_fractions(
                kind,
                query,
                &rank.operations,
                pricing.mix_step_passes.as_ref(),
            );
            for prefill in [true, false] {
                let fractions = prefill_fractions
                    .iter()
                    .map(|&fraction| if prefill { fraction } else { 1.0 - fraction });
                if let Some(phase) = self.operation_part_evidence(&rank.operations, fractions)? {
                    evidence.push((Self::corrected_phase(Arc::new(phase), correction)?, prefill));
                }
            }
        }
        Ok(evidence)
    }

    /// [`Self::rank_operation_evidence`] of a native iteration whose ranks
    /// report their operations in the detail `capture` selects: the
    /// diagnostics while they are captured, else operation values. The same
    /// operations and values form the same phase evidence; values form it
    /// under name lists that every rank with the same operations shares, a
    /// pure rank's or each part of a mixed rank's. A rank's evidence before
    /// correction depends only on its query, so it is built once per query.
    fn native_rank_evidence(
        &self,
        pricing: &IterationPricing,
        iteration: &crate::perfmodel::fpm::NativeIterationEstimate,
        queries: &[RankQuery],
        capture: bool,
    ) -> Result<Vec<(Arc<ValidatedTimingPhase>, bool)>> {
        ensure!(
            iteration.ranks.len() == queries.len(),
            "canonical model did not report every rank of an operation-graph iteration"
        );
        let correction = iteration.correction_factor;
        let mut evidence = Vec::with_capacity(queries.len());
        for (index, (rank, query)) in iteration.ranks.iter().zip(queries).enumerate() {
            let Some(kind) = IterationKind::of(query) else {
                continue;
            };
            let key = (*query, capture);
            let parts = match pricing.rank_evidence.get(&key) {
                Some(parts) => parts,
                None => {
                    let parts = Arc::new(if capture {
                        self.rank_diagnostics_parts(pricing, index, kind, query, rank)?
                    } else {
                        Self::rank_value_parts(pricing, index, kind, query, rank)?
                    });
                    pricing.rank_evidence.insert(key, Arc::clone(&parts));
                    parts
                }
            };
            for (phase, prefill) in parts.iter() {
                evidence.push((
                    Self::corrected_phase(Arc::clone(phase), correction)?,
                    *prefill,
                ));
            }
        }
        Ok(evidence)
    }

    /// One working rank's uncorrected evidence from its diagnostics, as
    /// [`Self::rank_operation_evidence`] builds it.
    fn rank_diagnostics_parts(
        &self,
        pricing: &IterationPricing,
        index: usize,
        kind: IterationKind,
        query: &RankQuery,
        rank: &RankPricing,
    ) -> Result<Vec<(Arc<ValidatedTimingPhase>, bool)>> {
        let operations = rank.diagnostics.as_deref().unwrap_or_default();
        ensure!(
            !operations.is_empty(),
            "canonical model reported no operations for rank {index}'s work"
        );
        let prefill_fractions =
            Self::prefill_fractions(kind, query, operations, pricing.mix_step_passes.as_ref());
        let mut parts = Vec::with_capacity(2);
        for prefill in [true, false] {
            let fractions = prefill_fractions
                .iter()
                .map(|&fraction| if prefill { fraction } else { 1.0 - fraction });
            if let Some(phase) = self.operation_part_evidence(operations, fractions)? {
                parts.push((Arc::new(phase), prefill));
            }
        }
        Ok(parts)
    }

    /// One working rank's uncorrected evidence from its operation values:
    /// the operations and values of [`Self::rank_diagnostics_parts`], without
    /// diagnostics.
    fn rank_value_parts(
        pricing: &IterationPricing,
        index: usize,
        kind: IterationKind,
        query: &RankQuery,
        rank: &RankPricing,
    ) -> Result<Vec<(Arc<ValidatedTimingPhase>, bool)>> {
        let operations = rank
            .operations
            .as_ref()
            .filter(|operations| !operations.values.is_empty());
        let Some(operations) = operations else {
            bail!("canonical model reported no operations for rank {index}'s work");
        };
        let value = |&(latency_ms, energy_wms, source): &PerOpSlotValue, fraction: f64| {
            Ok((
                latency_ms * fraction,
                Some(energy_wms * fraction),
                TimingEvidenceSource::from_native(source),
            ))
        };
        if kind != IterationKind::Mixed {
            // A pure rank is wholly its phase: every fraction is 1.
            let phase = ValidatedTimingPhase::from_slot_values(
                &operations.names,
                operations.values.iter().map(|value| {
                    let (latency_ms, energy_wms, source) = *value;
                    Ok((
                        latency_ms,
                        Some(energy_wms),
                        TimingEvidenceSource::from_native(source),
                    ))
                }),
            )?;
            return Ok(vec![(Arc::new(phase), kind == IterationKind::Prefill)]);
        }
        // The mix step's own token counts, as for diagnostics.
        let ctx_tokens = f64::from(query[1]);
        let token_share = ctx_tokens / (ctx_tokens + f64::from(query[3]));
        let split = pricing.mixed_split(&operations.names)?;
        let mut parts = Vec::with_capacity(2);
        for (prefill, part) in [(true, &split.prefill), (false, &split.decode)] {
            let fraction = |index: usize| {
                let fraction = split.prefill_fraction(index, token_share);
                if prefill { fraction } else { 1.0 - fraction }
            };
            // The operations with a positive share, as diagnostics keep.
            let mut included = (0..operations.values.len()).filter(|&index| fraction(index) > 0.0);
            let shared = part
                .indices
                .iter()
                .all(|&index| included.next() == Some(index))
                && included.next().is_none();
            let phase = if shared {
                ValidatedTimingPhase::from_slot_values(
                    &part.names,
                    part.indices
                        .iter()
                        .map(|&index| value(&operations.values[index], fraction(index))),
                )?
            } else {
                // A share at 0 or 1 leaves other operations out.
                let indices: Vec<_> = (0..operations.values.len())
                    .filter(|&index| fraction(index) > 0.0)
                    .collect();
                if indices.is_empty() {
                    continue;
                }
                let names = indices
                    .iter()
                    .map(|&index| operations.names[index].clone())
                    .collect();
                ValidatedTimingPhase::from_slot_values(
                    &names,
                    indices
                        .iter()
                        .map(|&index| value(&operations.values[index], fraction(index))),
                )?
            };
            parts.push((Arc::new(phase), prefill));
        }
        Ok(parts)
    }

    /// The prefill fraction of each of a working rank's `operations`; see
    /// [`Self::rank_operation_evidence`].
    fn prefill_fractions(
        kind: IterationKind,
        query: &RankQuery,
        operations: &[crate::perfmodel::engine::diagnostics::StaticOperationDiagnostics],
        passes: Option<&std::collections::HashMap<String, MixedPass>>,
    ) -> Vec<f64> {
        let token_share = match kind {
            IterationKind::Prefill => 1.0,
            IterationKind::Decode => 0.0,
            // The mix step's own token counts: the scheduled prefill tokens
            // and one token per packed decode request.
            IterationKind::Mixed => {
                let ctx_tokens = f64::from(query[1]);
                ctx_tokens / (ctx_tokens + f64::from(query[3]))
            }
        };
        let by_pass = (kind == IterationKind::Mixed)
            .then_some(passes)
            .flatten()
            .and_then(|passes| {
                operations
                    .iter()
                    .map(|operation| {
                        passes.get(&operation.name).map(|pass| match pass {
                            MixedPass::ContextAttention => 1.0,
                            MixedPass::DecodeAttention => 0.0,
                            MixedPass::SharedNonAttention => token_share,
                        })
                    })
                    .collect::<Option<Vec<_>>>()
            });
        by_pass.unwrap_or_else(|| vec![token_share; operations.len()])
    }

    /// Phase evidence of a part of name-folded `operations`: each operation's
    /// latency, energy and SOL times its fraction, and an operation with none
    /// left out. `None` when no operation has a part. Details are kept only
    /// while diagnostics are captured, as for phase evidence.
    fn operation_part_evidence(
        &self,
        operations: &[crate::perfmodel::engine::diagnostics::StaticOperationDiagnostics],
        fractions: impl Iterator<Item = f64>,
    ) -> Result<Option<ValidatedTimingPhase>> {
        let operations = operations
            .iter()
            .zip(fractions)
            .filter(|(_, fraction)| *fraction > 0.0)
            .map(|(entry, fraction)| {
                let mut operation = TimingOperationEvidence::new(
                    entry.name.clone(),
                    entry.latency_ms * fraction,
                    Some(entry.energy_wms * fraction),
                    TimingEvidenceSource::from_provider(entry.source.clone()),
                )?;
                if self.diagnostic_model.is_some() {
                    let mut details = entry.details.clone();
                    if let Some(sol) = &mut details.sol {
                        sol.latency_ms *= fraction;
                        sol.math_ms *= fraction;
                        sol.memory_ms *= fraction;
                    }
                    operation.details = Some(details);
                }
                Ok(operation)
            })
            .collect::<Result<Vec<_>>>()?;
        if operations.is_empty() {
            return Ok(None);
        }
        // Rank operations are name-folded in first-encounter order.
        ValidatedTimingPhase::from_name_folded_operations(operations).map(Some)
    }

    /// Record one invocation of a priced iteration: its operation evidence,
    /// and, while diagnostics are captured, every working rank's whole-model
    /// FPM estimate under the iteration's correction, as each rank recorded
    /// its own estimate when it priced its own phases.
    fn record_iteration(&self, priced: &PricedIteration, queries: &[RankQuery]) -> Result<()> {
        for (phase, prefill) in &priced.evidence {
            self.record_evidence(phase, *prefill)?;
        }
        let Some(estimate) = priced
            .estimate
            .as_ref()
            .filter(|_| self.diagnostic_model.is_some())
        else {
            return Ok(());
        };
        ensure!(
            estimate.ranks.len() == queries.len(),
            "canonical model did not report every rank of a whole-model FPM iteration"
        );
        let correction = estimate.correction_factor;
        let mut iterations = self
            .fpm_iteration_evidence
            .lock()
            .map_err(|_| anyhow!("FPM evidence accumulator was poisoned"))?;
        for (rank, query) in estimate.ranks.iter().zip(queries) {
            let Some(kind) = IterationKind::of(query) else {
                continue;
            };
            // As for a phase estimate, empty work runs no lookup, while a zero
            // marginal decode after real lookups remains evidence.
            if rank.latency_ms <= 0.0 && rank.decode_baseline_ms.is_none() {
                continue;
            }
            let key = (kind, *query, correction.map_or(0, f64::to_bits));
            let entry = iterations
                .entry(key)
                .or_insert_with(|| crate::FpmEstimateEvidence {
                    estimate: rank_estimate(rank, correction),
                    count: 0,
                    latency_scale: 1.0,
                });
            entry.count = entry
                .count
                .checked_add(1)
                .context("FPM evidence invocation count overflow")?;
        }
        Ok(())
    }
}

/// One rank of a multi-rank whole-model FPM estimate as its own estimate:
/// the rank's native latency under the iteration's `correction`, as a
/// one-rank query of the same work under that correction reports it.
fn rank_estimate(
    rank: &crate::FpmRankEstimate,
    correction: Option<f64>,
) -> crate::ForwardPassEstimate {
    let native = rank.latency_ms;
    crate::ForwardPassEstimate {
        unavailable_reason: None,
        latency_ms: Some(correction.map_or(native, |factor| native * factor)),
        native_latency_ms: Some(native),
        correction_factor: correction,
        max_rank: (native > 0.0).then_some(0),
        ranks: vec![crate::FpmRankEstimate {
            rank: 0,
            ..rank.clone()
        }],
    }
}

/// Past KV tokens of a decode step, excluding each request's current token.
fn total_past_kv_tokens(
    batch_size: usize,
    active_kv_tokens: usize,
    total_kv_tokens: usize,
) -> Result<usize> {
    Ok(active_kv_tokens
        .checked_sub(batch_size)
        .context("active decode tokens must include one current token per request")?
        .min(total_kv_tokens))
}

/// A static FPM estimate with a tuned model's online correction applied.
fn corrected_estimate(
    mut estimate: crate::ForwardPassEstimate,
    correction: f64,
) -> crate::ForwardPassEstimate {
    if correction != 1.0 {
        estimate.correction_factor = Some(correction);
        estimate.latency_ms = estimate.native_latency_ms.map(|native| native * correction);
    }
    estimate
}

/// Evidence of one native phase whose name-folded value `k` is named
/// `names[k]`, which every pass of a compiled phase shares.
fn phase_evidence_from_native_slots(
    names: &Arc<[String]>,
    values: Vec<crate::perfmodel::engine::runtime::PerOpSlotValue>,
) -> Result<ValidatedTimingPhase> {
    ValidatedTimingPhase::from_slot_values(
        names,
        names
            .iter()
            .zip(values)
            .map(|(name, (latency_ms, energy_wms, source))| {
                ensure!(
                    energy_wms.is_finite() && energy_wms >= 0.0,
                    "AIC operation {name:?} returned invalid energy {energy_wms}W-ms"
                );
                Ok((
                    latency_ms,
                    Some(energy_wms),
                    TimingEvidenceSource::from_native(source),
                ))
            }),
    )
}

fn phase_evidence_from_native_entries(
    entries: Vec<crate::perfmodel::engine::PerOpValue>,
) -> Result<ValidatedTimingPhase> {
    let (names, values): (Vec<_>, Vec<_>) = entries
        .into_iter()
        .map(|(name, latency_ms, energy_wms, source)| (name, (latency_ms, energy_wms, source)))
        .unzip();
    phase_evidence_from_native_slots(&names.into(), values)
}

#[cfg(test)]
fn phase_evidence_from_entries(
    entries: Vec<(String, f64, f64, String)>,
) -> Result<ValidatedTimingPhase> {
    let operations = entries
        .into_iter()
        .map(|(name, latency_ms, energy_wms, source)| {
            ensure!(
                energy_wms.is_finite() && energy_wms >= 0.0,
                "AIC operation {name:?} returned invalid energy {energy_wms}W-ms"
            );
            TimingOperationEvidence::new(
                name,
                latency_ms,
                Some(energy_wms),
                TimingEvidenceSource::from_provider(source),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ValidatedTimingPhase::from_operations(operations)
}

impl TimingModel for AicTimingModel {
    /// Price an attention-DP group pass with one canonical
    /// `ForwardPassMetrics` query, one entry per rank. `None` where the query
    /// cannot represent the pass: a measured SD-round profile with more than
    /// one rank, an operation graph whose measured scopes cannot compose a
    /// mix step, a scheduler at another speculative depth, or totals beyond
    /// the query's range.
    fn predict_iteration_ms(&self, ranks: &[RankWork]) -> Result<Option<IterationMs>> {
        let Some(pricing) = &self.iteration else {
            return Ok(None);
        };
        let mut queries = Vec::with_capacity(ranks.len());
        for rank in ranks {
            let query = match pricing.query {
                IterationQuery::Native(dispatch) => Self::rank_query(dispatch, rank)?,
                IterationQuery::Fitted { .. } => Some(Self::fitted_rank_query(rank)?),
            };
            let Some(query) = query else {
                return Ok(None);
            };
            queries.push(query);
        }
        if queries
            .iter()
            .all(|query| IterationKind::of(query).is_none())
        {
            // An iteration without model work computes nothing.
            return Ok(Some(IterationMs { total_ms: 0.0 }));
        }
        let mixed_rank = queries
            .iter()
            .any(|query| IterationKind::of(query) == Some(IterationKind::Mixed));
        match pricing.query {
            // Measured rounds stay outside attention DP; their mixed rank is
            // the prefill plus a full decode round.
            IterationQuery::Native(IterationDispatch::WholeModel { sd_round: true })
                if queries.len() != 1 =>
            {
                return Ok(None);
            }
            IterationQuery::Native(IterationDispatch::Operations { mixed: false, .. })
                if mixed_rank =>
            {
                return Ok(None);
            }
            _ => {}
        }
        let capture = self.diagnostic_model.is_some();
        let whole_model = matches!(
            pricing.query,
            IterationQuery::Native(IterationDispatch::WholeModel { .. })
        );
        let cached = pricing
            .cache
            .as_ref()
            .and_then(|cache| cache.get(&queries))
            .filter(|priced| !(capture && whole_model && priced.estimate.is_none()));
        let (priced, hit) = match cached {
            Some(priced) => (priced, true),
            None => (
                Arc::new(self.price_iteration(pricing, &queries, ranks, capture)?),
                false,
            ),
        };
        self.record_iteration(&priced, &queries)?;
        let iteration = IterationMs {
            total_ms: priced.total_ms,
        };
        if let Some(cache) = pricing.cache.as_ref().filter(|_| !hit) {
            cache.insert(queries, priced);
        }
        Ok(Some(iteration))
    }

    fn predict_vision_ms(&self, shapes: &[VisionShape]) -> Result<Option<f64>> {
        if self.vision.is_none() {
            return Ok(None);
        }
        let mut latency_ms = 0.0;
        for shape in shapes {
            let phase = self.vision_phase(*shape)?;
            latency_ms += phase.latency_ms();
            // The encoder runs inside the prefill forward; its operations join
            // that phase's evidence under their own `encoder_*` names.
            self.record_evidence(&phase, true)?;
        }
        Ok(Some(latency_ms))
    }

    fn validate_prefill_batch(&self, requests: &[(usize, usize)]) -> Result<()> {
        if self.decoder_replay {
            ensure!(
                requests.windows(2).all(|pair| pair[0] == pair[1]),
                "decoder replay requires identical per-request new-token and cached-prefix lengths; \
                 heterogeneous prefill cannot be represented by the mean-based replay timing API"
            );
        }
        Ok(())
    }

    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        self.require_phase_pricing()?;
        let correction = if self.corrections.is_some() {
            let tokens = |per_request: usize| {
                per_request
                    .checked_mul(batch_size)
                    .context("prefill token total overflowed")
            };
            self.prefill_correction(
                batch_size,
                tokens(mean_isl.saturating_sub(mean_prefix))?,
                tokens(mean_prefix)?,
            )?
        } else {
            1.0
        };
        self.native_prefill_ms(batch_size, mean_isl, mean_prefix, correction)
    }

    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> Result<f64> {
        self.require_phase_pricing()?;
        let correction = self.prefill_correction(batch_size, new_tokens, prefix_tokens)?;
        if !self.use_fpm_decode_totals {
            // Operation graphs price per-request shapes.
            let (mean_isl, mean_prefix) =
                crate::engine::mean_prefill_shape(batch_size, new_tokens, prefix_tokens);
            return self.native_prefill_ms(batch_size, mean_isl, mean_prefix, correction);
        }
        let batch_size = checked_u32(batch_size, "prefill batch size")?;
        let total_prefill_tokens = checked_u32(new_tokens, "total prefill tokens")?;
        let total_kv_read_tokens = checked_u32(prefix_tokens, "total prefill KV tokens")?;
        if let Some(model) = &self.diagnostic_model {
            return self.record_fpm_estimate(
                (
                    FpmEvidencePhase::Prefill,
                    batch_size,
                    total_prefill_tokens,
                    total_kv_read_tokens,
                ),
                corrected_estimate(
                    model.static_prefill_detailed_total(
                        batch_size,
                        total_prefill_tokens,
                        total_kv_read_tokens,
                    )?,
                    correction,
                ),
            );
        }
        if let Some(model) = &self.fpm_model {
            return model
                .predict_prefill_latency_total(
                    batch_size,
                    total_prefill_tokens,
                    total_kv_read_tokens,
                )
                .map(|native| native * correction)
                .context("AIC prefill prediction failed");
        }
        Python::with_gil(|py| {
            self.engine
                .bind(py)
                .call_method1(
                    "predict_prefill_latency_total",
                    (batch_size, total_prefill_tokens, total_kv_read_tokens),
                )?
                .extract::<f64>()
        })
        .map(|native| native * correction)
        .map_err(|error| anyhow!("AIC prefill prediction failed: {error}"))
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        self.require_phase_pricing()?;
        if self.use_fpm_decode_totals {
            let total_past_kv_tokens =
                total_past_kv_tokens(batch_size, active_kv_tokens, total_kv_tokens)?;
            let correction = self.decode_correction(batch_size, || Ok(total_past_kv_tokens))?;
            let batch_size = checked_u32(batch_size, "decode batch size")?;
            let total_past_kv_tokens = checked_u32(total_past_kv_tokens, "total past KV tokens")?;
            // The detailed estimate applies the same correction itself.
            if let Some(model) = &self.diagnostic_model {
                let estimate =
                    model.estimate_forward_pass_detailed(&[crate::ForwardPassMetrics {
                        scheduled_requests: crate::ScheduledRequestMetrics {
                            num_decode_requests: batch_size,
                            sum_decode_kv_tokens: total_past_kv_tokens,
                            ..Default::default()
                        },
                        ..Default::default()
                    }])?;
                return self.record_fpm_estimate(
                    (
                        FpmEvidencePhase::Decode,
                        batch_size,
                        0,
                        total_past_kv_tokens,
                    ),
                    estimate,
                );
            }
            if let Some(model) = &self.fpm_model {
                return model
                    .predict_decode_latency_total(batch_size, total_past_kv_tokens)
                    .map(|native| native * correction)
                    .context("AIC decode prediction failed");
            }
            return Python::with_gil(|py| {
                self.engine
                    .bind(py)
                    .call_method1(
                        "predict_decode_latency_total",
                        (batch_size, total_past_kv_tokens),
                    )?
                    .extract::<f64>()
            })
            .map(|native| native * correction)
            .map_err(|error| anyhow!("AIC decode prediction failed: {error}"));
        }

        let correction = self.decode_correction(batch_size, || {
            total_past_kv_tokens(batch_size, active_kv_tokens, total_kv_tokens)
        })?;
        let batch_size = checked_u32(batch_size, "decode batch size")?;
        // Both scheduler implementations include the current input token in
        // their sequence length before sampling the next token. The op API's
        // isl is past KV; osl=2 adds the current token exactly once.
        let mean_past_kv = mean_context_length
            .checked_sub(1)
            .context("mean decode context must include the current input token")?;
        let mean_past_kv = checked_u32(mean_past_kv, "mean past KV length")?;
        let evidence = Self::corrected_phase(
            self.predict_phase_evidence(batch_size, mean_past_kv, 2, 0, "static_gen")?,
            correction,
        )?;
        let latency_ms = evidence.latency_ms();
        self.record_evidence(&evidence, false)?;
        Ok(latency_ms)
    }

    fn predict_mixed_pass_ms(
        &self,
        prefill: PrefillTotals,
        batch_size: usize,
        active_kv_tokens: usize,
        _mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<Option<MixedPassMs>> {
        self.require_phase_pricing()?;
        // Whole-model FPM prices a mixed pass as one forward: the prefill it
        // computes plus the latency its decode requests add, which does not
        // depend on that prefill. A tuned model corrects that forward as the
        // mixed iteration it is. Operation graphs keep separate passes here.
        if !self.use_fpm_decode_totals {
            return Ok(None);
        }
        let total_past_kv_tokens =
            total_past_kv_tokens(batch_size, active_kv_tokens, total_kv_tokens)?;
        let prefill_batch = checked_u32(prefill.batch_size, "prefill batch size")?;
        let prefill_tokens = checked_u32(prefill.new_tokens, "total prefill tokens")?;
        let prefill_kv_tokens = checked_u32(prefill.prefix_tokens, "total prefill KV tokens")?;
        let decode_batch = checked_u32(batch_size, "decode batch size")?;
        let decode_kv_tokens = checked_u32(total_past_kv_tokens, "total past KV tokens")?;
        // The decode share comes first: an engine without the composition
        // returns `None` before any lookup, so the scheduler's separate
        // prefill is never looked up twice.
        if let Some(model) = &self.diagnostic_model {
            let Some(decode) = model.mixed_decode_detailed_total(decode_batch, decode_kv_tokens)?
            else {
                return Ok(None);
            };
            let prefill_estimate = model.static_prefill_detailed_total(
                prefill_batch,
                prefill_tokens,
                prefill_kv_tokens,
            )?;
            let correction = self.mixed_correction(prefill, batch_size, total_past_kv_tokens)?;
            return Ok(Some(MixedPassMs {
                prefill_ms: self.record_fpm_estimate(
                    (
                        FpmEvidencePhase::Prefill,
                        prefill_batch,
                        prefill_tokens,
                        prefill_kv_tokens,
                    ),
                    corrected_estimate(prefill_estimate, correction),
                )?,
                decode_ms: self.record_fpm_estimate(
                    (
                        FpmEvidencePhase::MixedDecode,
                        decode_batch,
                        0,
                        decode_kv_tokens,
                    ),
                    corrected_estimate(decode, correction),
                )?,
            }));
        }
        let Some(model) = &self.fpm_model else {
            return Ok(None);
        };
        let Some(decode_ms) = model
            .predict_mixed_decode_latency_total(decode_batch, decode_kv_tokens)
            .context("AIC mixed decode prediction failed")?
        else {
            return Ok(None);
        };
        let prefill_ms = model
            .predict_prefill_latency_total(prefill_batch, prefill_tokens, prefill_kv_tokens)
            .context("AIC prefill prediction failed")?;
        let correction = self.mixed_correction(prefill, batch_size, total_past_kv_tokens)?;
        Ok(Some(MixedPassMs {
            prefill_ms: prefill_ms * correction,
            decode_ms: decode_ms * correction,
        }))
    }

    fn prices_phases(&self) -> bool {
        !self.is_fitted()
    }

    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        // Whole-model FPM, regression and learned estimators are latency-only.
        if self.use_fpm_decode_totals || self.is_fitted() {
            return None;
        }
        self.evidence
            .lock()
            .ok()
            .map(|evidence| evidence.snapshot())
    }

    fn fpm_evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        if !self.use_fpm_decode_totals || self.diagnostic_model.is_none() {
            return None;
        }
        let history = self.fpm_evidence.lock().ok()?;
        let iterations = self.fpm_iteration_evidence.lock().ok()?;
        let phase = |prefill: bool| {
            let fpm_estimates: Vec<_> = history
                .iter()
                .filter(|(key, _)| (key.0 == FpmEvidencePhase::Prefill) == prefill)
                .map(|(_, evidence)| evidence.clone())
                .chain(
                    iterations
                        .iter()
                        .filter(|(key, _)| (key.0 == IterationKind::Prefill) == prefill)
                        .map(|(_, evidence)| evidence.clone()),
                )
                .collect();
            if fpm_estimates.is_empty() {
                return TimingPhaseEvidence::default();
            }
            let latency_ms = fpm_estimates
                .iter()
                .map(|record| record.estimate.latency_ms.unwrap_or(0.0) * record.count as f64)
                .sum();
            let mut operation = TimingOperationEvidence::new(
                if prefill { "fpm_prefill" } else { "fpm_decode" },
                latency_ms,
                None,
                TimingEvidenceSource::Silicon,
            )
            .expect("finite native FPM evidence");
            operation.details = Some(crate::perfmodel::engine::diagnostics::OperationDetails {
                sol: None,
                sol_unavailable_reason: Some(
                    "whole-model FPM does not export an operation SOL decomposition".into(),
                ),
                fallbacks: Vec::new(),
                fpm_estimates,
            });
            TimingPhaseEvidence::from_operations(vec![operation])
        };
        Some(TimingEvidenceSummary {
            prefill: phase(true),
            decode: phase(false),
        })
    }

    fn reset_evidence(&self) -> Result<()> {
        self.fpm_evidence
            .lock()
            .map_err(|_| anyhow!("FPM evidence accumulator was poisoned"))?
            .clear();
        self.fpm_iteration_evidence
            .lock()
            .map_err(|_| anyhow!("FPM evidence accumulator was poisoned"))?
            .clear();
        *self
            .evidence
            .lock()
            .map_err(|_| anyhow!("AIC timing evidence accumulator was poisoned"))? =
            TimingEvidenceAccumulator::default();
        Ok(())
    }
}

/// Require the workload stores a regression or learned estimator needs to
/// price every iteration replay schedules for a `worker_type` role with
/// `attention_dp` ranks: pure prefill for a prefill worker, pure decode for a
/// decode worker, and both for an aggregated worker. An aggregated rank also
/// computes prefill and decode together when its backend's scheduler does
/// (see [`schedules_locally_mixed`]), and its ranks compute different kinds of
/// work when attention DP is known to be above one. Unknown attention DP adds
/// no store here; the role checks again with its engine's width.
fn require_iteration_stores(
    model: &ForwardPassPerfModel,
    worker_type: ForwardPassWorkerType,
    attention_dp: u32,
) -> Result<()> {
    use ForwardPassRegressionWorkloadKind::*;
    let provenance = model
        .provenance()
        .context("canonical estimator omitted construction provenance")?;
    let mut required = match worker_type {
        ForwardPassWorkerType::Prefill => vec![PurePrefill],
        ForwardPassWorkerType::Decode => vec![PureDecode],
        ForwardPassWorkerType::Aggregated => vec![PurePrefill, PureDecode],
    };
    if worker_type == ForwardPassWorkerType::Aggregated {
        if schedules_locally_mixed(&provenance.config.backend) {
            required.push(ContainsLocallyMixed);
        }
        if attention_dp > 1 && attention_dp != UNKNOWN_SIZE {
            required.push(CrossRankAggregated);
        }
    }
    let report = model.readiness_report();
    let missing = report
        .stores
        .iter()
        .filter(|store| required.contains(&store.workload_kind) && !store.ready)
        .map(|store| match store.not_ready_reason {
            Some(reason) => format!("{} ({})", wire_name(store.workload_kind), wire_name(reason)),
            None => wire_name(store.workload_kind),
        })
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "the {} estimator cannot price every iteration of a {} worker: workload store(s) {} not ready; \
         tune it with observed forward passes of that work, or restore a snapshot calibrated on it, before replay",
        wire_name(provenance.selected_estimation_mode),
        wire_name(worker_type),
        missing.join(", ")
    );
    Ok(())
}

/// Whether `backend`'s replay scheduler computes prefill and decode in one
/// rank pass. vLLM, and TensorRT-LLM on the shared vLLM core, schedule chunked
/// prefill beside running decodes. SGLang runs a formed prefill batch alone
/// (prefill and decode share a forward only with `--enable-mixed-chunk`,
/// which is not modeled), so its ranks never compute locally mixed work.
fn schedules_locally_mixed(backend: &crate::BackendKind) -> bool {
    !matches!(backend, crate::BackendKind::Sglang)
}

/// Require the unit `decode_speedup_ratio` a regression or learned role needs.
/// Replay prices a group pass with one iteration query only at a unit decode
/// speedup; at any other, each rank prices its own phases, which such an
/// estimator never does, so the role would fail at its first pass.
fn require_unit_decode_speedup(
    model: &ForwardPassPerfModel,
    decode_speedup_ratio: f64,
) -> Result<()> {
    ensure!(
        decode_speedup_ratio == 1.0,
        "{} replay requires decode_speedup_ratio=1.0, got {decode_speedup_ratio}: it prices only whole \
         iterations, and replay prices a pass as one iteration only at a unit decode speedup",
        model.provenance().map_or_else(
            || "regression or learned".to_owned(),
            |provenance| wire_name(provenance.selected_estimation_mode)
        )
    );
    Ok(())
}

/// The snake-case wire spelling of a unit enum value.
fn wire_name(value: impl serde::Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        other => format!("{other:?}"),
    }
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} {value} exceeds AIC's u32 limit"))
}

fn aic_capacity_kwargs<'py>(
    py: Python<'py>,
    config: &AicTimingConfig,
    role: &ReplayRoleConfig,
) -> PyResult<Bound<'py, PyDict>> {
    let (memory_fraction_kind, memory_fraction_value) = config
        .resolved_memory_fraction()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("backend_version", config.resolved_backend_version())?;
    kwargs.set_item("scheduler_block_size", role.rank.block_size)?;
    kwargs.set_item("max_num_tokens", role.rank.max_num_batched_tokens)?;
    kwargs.set_item("max_batch_size", role.rank.max_num_seqs)?;
    if config.fpm_profile.is_some() {
        kwargs.set_item("context_length", role.rank.max_model_len)?;
        kwargs.set_item(
            "worker_type",
            match config
                .worker_type
                .unwrap_or(ForwardPassWorkerType::Aggregated)
            {
                ForwardPassWorkerType::Prefill => "prefill",
                ForwardPassWorkerType::Decode => "decode",
                ForwardPassWorkerType::Aggregated => "aggregated",
            },
        )?;
    }
    kwargs.set_item("memory_fraction_kind", memory_fraction_kind)?;
    kwargs.set_item("memory_fraction_value", memory_fraction_value)?;
    kwargs.set_item("tp_size", config.tp)?;
    kwargs.set_item("pp_size", config.pp)?;
    kwargs.set_item("attention_dp_size", config.attention_dp)?;
    kwargs.set_item("moe_tp_size", config.moe_tp_size)?;
    kwargs.set_item("moe_ep_size", config.moe_ep_size)?;
    kwargs.set_item("cp_size", config.cp_size.unwrap_or(1))?;
    // Capacity estimation prices the 1/dcp KV stripe; unrecorded DCP is 1.
    kwargs.set_item("dcp_size", config.dcp.unwrap_or(1))?;
    kwargs.set_item("gemm_quant_mode", config.gemm_dtype.as_deref())?;
    kwargs.set_item("moe_quant_mode", config.moe_dtype.as_deref())?;
    kwargs.set_item("fmha_quant_mode", config.fmha_dtype.as_deref())?;
    kwargs.set_item("kvcache_quant_mode", config.kv_cache_dtype.as_deref())?;
    kwargs.set_item("comm_quant_mode", config.comm_dtype.as_deref())?;
    kwargs.set_item("moe_backend", config.moe_backend.as_deref())?;
    kwargs.set_item("attention_backend", config.attention_backend.as_deref())?;
    kwargs.set_item("enable_eplb", config.enable_eplb)?;
    kwargs.set_item("wideep_num_slots", config.wideep_num_slots)?;
    kwargs.set_item(
        "fpm_profile",
        config
            .fpm_profile
            .as_ref()
            .map(|profile| serde_json::to_string(profile).expect("JSON profile")),
    )?;
    // A rank hosting the vision encoder keeps its weights and embedding
    // cache outside the KV pool.
    kwargs.set_item("colocated_encoder", role.rank.vision)?;
    kwargs.set_item(
        "encoder_parallel",
        config.encoder_parallel.map(crate::EncoderParallel::as_str),
    )?;
    kwargs.set_item(
        "reserved_bytes",
        if role.rank.vision {
            role.rank.sglang.vlm_cache_bytes
        } else {
            0
        },
    )?;
    kwargs.set_item(
        "cuda_graph_reserved_bytes",
        config.cuda_graph_reserved_bytes,
    )?;
    // Legacy analytical capacity still omits NextN. Profile resources have
    // already been validated by materialize_aic_capacity; nonzero depth here
    // identifies measured SD memory, whose recorded pool must not be re-priced.
    if config.fpm_profile.is_some() && config.nextn > 0 {
        kwargs.set_item("nextn", config.nextn)?;
    }
    kwargs.set_item("systems_path", config.systems_path.as_deref())?;
    Ok(kwargs)
}

fn estimate_aic_num_gpu_blocks(config: &AicTimingConfig, role: &ReplayRoleConfig) -> Result<usize> {
    Python::with_gil(|py| -> PyResult<usize> {
        let memory = PyModule::import(py, "aisimulate_core.sdk.memory")?;
        let kwargs = aic_capacity_kwargs(py, config, role)?;
        memory
            .getattr("estimate_num_gpu_blocks")?
            .call(
                (
                    config.model.as_str(),
                    config.system.as_str(),
                    config.backend.as_str(),
                ),
                Some(&kwargs),
            )?
            .extract()
    })
    .map_err(|error| anyhow!("AIC KV-cache capacity estimation failed: {error}"))
}

fn estimate_aic_grouped_budget(
    config: &AicTimingConfig,
    role: &ReplayRoleConfig,
) -> Result<crate::perfmodel::fpm::FpmCacheBudget> {
    let serialized = Python::with_gil(|py| -> PyResult<String> {
        let memory = PyModule::import(py, "aisimulate_core.sdk.memory")?;
        let kwargs = aic_capacity_kwargs(py, config, role)?;
        kwargs.del_item("scheduler_block_size")?;
        kwargs.set_item("context_length", role.rank.max_model_len)?;
        let budget = memory.getattr("estimate_kv_cache")?.call(
            (
                config.model.as_str(),
                config.system.as_str(),
                config.backend.as_str(),
            ),
            Some(&kwargs),
        )?;
        PyModule::import(py, "json")?
            .call_method1("dumps", (budget,))?
            .extract()
    })
    .map_err(|error| anyhow!("AIC grouped-cache budget estimation failed: {error}"))?;
    serde_json::from_str(&serialized).context("invalid AIC grouped-cache budget")
}

fn materialize_aic_capacity(
    config: &AicTimingConfig,
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
    estimate: impl FnOnce(&AicTimingConfig, &ReplayRoleConfig) -> Result<usize>,
) -> Result<()> {
    ensure!(
        capacity_is_explicit
            || config
                .speculation
                .as_ref()
                .is_none_or(|spec| matches!(spec.kind.as_str(), "none" | "ngram")),
        "speculative replay requires explicit num_gpu_blocks including draft memory"
    );
    config.validate_parallel_shape()?;
    let engine_backend = match role.rank.backend {
        Backend::Vllm => "vllm",
        Backend::Sglang => "sglang",
        Backend::Trtllm => "trtllm",
    };
    ensure!(
        config.backend == engine_backend,
        "AIC backend {:?} does not match engine backend {engine_backend:?}",
        config.backend
    );
    // A size the timing identity records as unknown takes the engine's.
    ensure!(
        config.tp == UNKNOWN_SIZE || config.tp == role.tensor_parallel_size,
        "AIC tp={} does not match engine tensor_parallel_size={}",
        config.tp,
        role.tensor_parallel_size
    );
    ensure!(
        config.cp_size.unwrap_or(1) == role.prefill_context_parallel_size,
        "AIC cp_size does not match engine prefill_context_parallel_size={}",
        role.prefill_context_parallel_size
    );
    ensure!(
        config.attention_dp == UNKNOWN_SIZE || config.attention_dp == role.dp_size,
        "AIC attention_dp={} does not match engine dp_size={}",
        config.attention_dp,
        role.dp_size
    );
    ensure!(
        config
            .kv_block_size
            .is_none_or(|block_size| block_size as usize == role.rank.block_size),
        "AIC kv_block_size does not match engine block_size={}",
        role.rank.block_size
    );
    let engine_nextn = role.rank.aic_nextn.unwrap_or(0);
    let timing_depth = config.speculative_depth()?;
    ensure!(
        timing_depth as usize == engine_nextn,
        "AIC speculative depth={timing_depth} does not match engine aic_nextn={engine_nextn}"
    );
    let resources = if config.backend == "vllm" && config.pp == 1 {
        config
            .estimator_request(
                config
                    .worker_type
                    .unwrap_or(ForwardPassWorkerType::Aggregated),
            )?
            .fpm_resources()?
    } else {
        None
    };
    if let Some(resources) = &resources {
        resources.require_memory()?;
        ensure!(
            resources.runtime_memory.is_none() || !capacity_is_explicit,
            "runtime FPM memory cannot use fixed num_gpu_blocks; use the recorded cache allocation"
        );
    }
    if resources.as_ref().is_some_and(|resources| {
        resources.cache_layout == crate::perfmodel::fpm::FpmCacheLayout::Grouped
    }) {
        ensure!(
            !capacity_is_explicit,
            "grouped FPM cache cannot use fixed num_gpu_blocks; use the profile byte budget"
        );
        let budget = estimate_aic_grouped_budget(config, role)?;
        ensure!(
            role.rank.kv_cache_groups.is_empty()
                || role.rank.kv_cache_groups == budget.cache_groups,
            "engine cache groups conflict with the canonical FPM profile"
        );
        ensure!(
            role.rank
                .kv_cache_capacity_bytes
                .is_none_or(|bytes| bytes == budget.total_kv_size_bytes),
            "engine cache byte capacity conflicts with the canonical FPM resource budget"
        );
        role.rank.kv_cache_groups = budget.cache_groups;
        role.rank.kv_cache_capacity_bytes = Some(budget.total_kv_size_bytes);
        // Keep the historical positive field for EngineConfig compatibility.
        // The grouped allocator exclusively consumes groups and byte capacity.
        role.rank.validate()?;
        return Ok(());
    }
    ensure!(
        role.rank.kv_cache_groups.is_empty() && role.rank.kv_cache_capacity_bytes.is_none(),
        "grouped cache allocation requires matching canonical FPM profile resources"
    );
    if capacity_is_explicit || role.rank.state_cache.is_some() {
        return Ok(());
    }
    if let Some(blocks) = recorded_kv_blocks(config, role)? {
        role.rank.num_gpu_blocks = blocks;
        return Ok(());
    }
    let blocks = estimate(config, role)?;
    ensure!(blocks > 0, "AIC estimated zero KV-cache blocks");
    role.rank.num_gpu_blocks = blocks;
    Ok(())
}

/// The KV capacity a regression or learned estimator's identity records for
/// its deployment, `deployment_limits.max_kv_tokens` in whole scheduler
/// blocks; `None` leaves capacity to the AIC memory estimate. That estimate
/// needs the complete identity, so a fitted identity with unknown facts and
/// no recorded capacity is an error.
fn recorded_kv_blocks(config: &AicTimingConfig, role: &ReplayRoleConfig) -> Result<Option<usize>> {
    if !matches!(
        config.estimation_mode,
        Some(EstimationMode::FpmRegression | EstimationMode::FpmLearned)
    ) {
        return Ok(None);
    }
    let max_kv_tokens = config
        .deployment_limits
        .and_then(|limits| limits.max_kv_tokens.known());
    let Some(max_kv_tokens) = max_kv_tokens else {
        let unknown = config
            .estimator_request(
                config
                    .worker_type
                    .unwrap_or(ForwardPassWorkerType::Aggregated),
            )?
            .unknown_identity_facts();
        ensure!(
            unknown.is_empty(),
            "AIC cannot estimate KV-cache capacity for an identity with unknown {}; set num_gpu_blocks \
             or record deployment_limits.max_kv_tokens",
            unknown.join(", ")
        );
        return Ok(None);
    };
    let blocks = usize::try_from(max_kv_tokens)? / role.rank.block_size;
    ensure!(
        blocks > 0,
        "deployment_limits.max_kv_tokens={max_kv_tokens} does not cover one scheduler block of {} tokens",
        role.rank.block_size
    );
    Ok(Some(blocks))
}

fn cap_role_capacity_to_fpm_decode_domain(
    role: &mut ReplayRoleConfig,
    decode_kv_ceiling: Option<u32>,
    capacity_is_recorded: bool,
) -> Result<()> {
    // A timing-table token ceiling is not a physical grouped-cache budget.
    // Queries retain their logical contexts and report uncovered points normally.
    if !role.rank.kv_cache_groups.is_empty() {
        return Ok(());
    }
    let Some(decode_kv_ceiling) = decode_kv_ceiling else {
        return Ok(());
    };
    if capacity_is_recorded || role.rank.state_cache.is_some() {
        return Ok(());
    }
    let covered_blocks = decode_kv_ceiling as usize / role.rank.block_size;
    ensure!(
        covered_blocks > 0,
        "FPM decode KV ceiling {decode_kv_ceiling} does not cover one scheduler block of {} tokens",
        role.rank.block_size
    );
    role.rank.num_gpu_blocks = role.rank.num_gpu_blocks.min(covered_blocks);
    Ok(())
}

fn aggregated_role(engine: &ReplayEngineConfig) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: engine.dp_size,
        tensor_parallel_size: engine.tensor_parallel_size,
        prefill_context_parallel_size: engine.prefill_context_parallel_size,
        num_gpu_blocks_is_explicit: engine.num_gpu_blocks_is_explicit,
        rank: engine.rank.clone(),
    }
}

fn role_capacity_is_explicit(engine_value: &serde_json::Value, role: Option<&str>) -> bool {
    let role_rank = role
        .and_then(|role| engine_value.get(role))
        .and_then(|role| role.get("rank"));
    role_rank
        .or_else(|| engine_value.get("rank"))
        .and_then(serde_json::Value::as_object)
        .is_some_and(|rank| {
            rank.contains_key("num_gpu_blocks")
                || rank
                    .get("state_cache")
                    .is_some_and(|value| !value.is_null())
        })
}

/// Resolve the encoder pool's timing model the way a rank's is resolved: the
/// canonical model compiled with its vision tower at the pool's tensor width.
/// The pool never shares a rank's model, so the spec must name one; built-in
/// models return `None` and the Replayer then rejects the spec.
fn resolve_encoder_timing(
    encoder: &EncoderSpec,
    capture_performance_diagnostics: bool,
    coverage_sources: &mut Vec<(ForwardPassWorkerType, Arc<ForwardPassPerfModel>)>,
) -> Result<Option<Arc<dyn TimingModel>>> {
    let timing_model = encoder
        .timing_model
        .clone()
        .context("encoder pool requires a timing_model")?;
    let TimingModelConfig::External { provider, config } = timing_model else {
        return Ok(None);
    };
    ensure!(
        provider == "aic",
        "native timing provider {provider:?} is not installed; only \"aic\" is \
         available in the AISimulate runtime"
    );
    let mut config: AicTimingConfig =
        serde_json::from_value(config).context("invalid encoder AIC timing configuration")?;
    ensure!(
        usize::try_from(config.tp).ok() == Some(encoder.gpus_per_instance),
        "encoder gpus_per_instance must equal the timing model's tensor-parallel width"
    );
    let mut timing = AicTimingModel::build(&mut config, ForwardPassWorkerType::Prefill, true)?;
    // An encoder pool prices only vision batches.
    timing.iteration = None;
    if let Some(model) = &timing.fpm_model
        && model.fpm_query_coverage()?.is_some()
    {
        coverage_sources.push((ForwardPassWorkerType::Prefill, Arc::clone(model)));
    }
    if !capture_performance_diagnostics {
        timing.diagnostic_model = None;
    }
    Ok(Some(Arc::new(timing)))
}

// Read the descriptor after canonical construction has pinned auto/fallback selection.
// This does not construct another model or enable operation-level diagnostics.
fn timing_provenance(timing: &TimingModelConfig) -> serde_json::Value {
    match timing {
        TimingModelConfig::External { provider, config } => {
            let estimator = config
                .get("estimation_mode")
                .and_then(|value| value.as_str());
            let is_fpm = matches!(estimator, Some("fpm_interpolation" | "fpm_regression"));
            let profile = config.get("fpm_profile").filter(|_| is_fpm);
            let data_version = if estimator == Some("op_level") {
                config
                    .get("backend_version")
                    .and_then(|value| value.as_str())
            } else {
                profile
                    .and_then(|value| value.get("model_revision"))
                    .and_then(|value| value.as_str())
            };
            let data_identity = profile.map(|profile| {
                serde_json::json!({
                    "model": profile.get("model"),
                    "model_revision": profile.get("model_revision"),
                })
            });
            serde_json::json!({
                "estimator": estimator,
                "timing_source": provider,
                "data_version": data_version,
                "data_identity": data_identity,
                "data_version_unavailable_reason": data_version.is_none().then_some(
                    "selected estimator does not expose a data version"
                ),
            })
        }
        TimingModelConfig::Fixed { .. } | TimingModelConfig::Polynomial => {
            let estimator = if matches!(timing, TimingModelConfig::Fixed { .. }) {
                "fixed"
            } else {
                "polynomial"
            };
            serde_json::json!({
                "estimator": estimator,
                "timing_source": if estimator == "polynomial" { "built_in" } else { "configured" },
                "data_version": null,
                "data_version_unavailable_reason": if estimator == "polynomial" {
                    "built-in timing does not use performance data"
                } else {
                    "configured timing does not use performance data"
                },
            })
        }
    }
}

/// vLLM turns async scheduling off by default for prompt-lookup (ngram)
/// drafts and rejects an explicit request for it.
fn resolve_ngram_async_scheduling(rank: &mut EngineConfig, speculation_kind: &str) -> Result<()> {
    if speculation_kind == "ngram" && rank.backend == Backend::Vllm {
        ensure!(
            rank.async_scheduling != Some(true),
            "vLLM does not support async scheduling with ngram speculative decoding"
        );
        rank.async_scheduling = Some(false);
    }
    Ok(())
}

fn resolve_role_timing(
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
    worker_type: ForwardPassWorkerType,
    capture_performance_diagnostics: bool,
    coverage_sources: &mut Vec<(ForwardPassWorkerType, Arc<ForwardPassPerfModel>)>,
) -> Result<Option<Arc<dyn TimingModel>>> {
    let TimingModelConfig::External { provider, config } = role.rank.timing_model.clone() else {
        return Ok(None);
    };
    ensure!(
        provider == "aic",
        "native timing provider {provider:?} is not installed; only \"aic\" is \
         available in the AISimulate runtime"
    );
    let mut config: AicTimingConfig =
        serde_json::from_value(config).context("invalid AIC timing provider configuration")?;
    let resources = if config.backend == "vllm" && config.pp == 1 {
        config.estimator_request(worker_type)?.fpm_resources()?
    } else {
        None
    };
    if let Some(resources) = &resources {
        resources.require_memory()?;
    }
    let mut timing = AicTimingModel::build(&mut config, worker_type, role.rank.vision)?;
    if let Some(pricing) = timing.iteration.as_ref().filter(|_| timing.is_fitted()) {
        // The identity may not record attention DP; the engine's width decides.
        require_iteration_stores(&pricing.model, worker_type, role.dp_size)?;
        require_unit_decode_speedup(&pricing.model, role.rank.decode_speedup_ratio)?;
    }
    if let Some(speculation) = &config.speculation {
        ensure!(
            speculation.kind == "ngram"
                || role
                    .rank
                    .aic_nextn_accept_rates
                    .as_deref()
                    .is_some_and(|rates| !rates.trim().is_empty()),
            "generic speculative replay requires explicit aic_nextn_accept_rates; \
             configure expected_accepted_tokens in engine.speculation"
        );
        let depth = config.speculative_depth()? as usize;
        ensure!(
            role.rank.aic_nextn.is_none_or(|value| value == depth),
            "aic_nextn conflicts with resolved speculative depth {depth}"
        );
        role.rank.aic_nextn = (depth > 0).then_some(depth);
        resolve_ngram_async_scheduling(&mut role.rank, &speculation.kind)?;
    }
    if let Some(model) = &timing.fpm_model
        && model.fpm_query_coverage()?.is_some()
    {
        coverage_sources.push((worker_type, Arc::clone(model)));
    }
    if !capture_performance_diagnostics {
        timing.diagnostic_model = None;
    }
    materialize_aic_capacity(
        &config,
        role,
        capacity_is_explicit,
        estimate_aic_num_gpu_blocks,
    )?;
    cap_role_capacity_to_fpm_decode_domain(
        role,
        timing.fpm_decode_kv_ceiling,
        capacity_is_explicit
            || resources
                .as_ref()
                .is_some_and(|resources| resources.runtime_memory.is_some()),
    )?;
    let mut resolved = serde_json::to_value(config.estimator_request(worker_type)?)?;
    if let Some(object) = resolved.as_object_mut() {
        for (name, value) in [
            ("gpu_memory_utilization", config.gpu_memory_utilization),
            ("mem_fraction_static", config.mem_fraction_static),
            ("free_gpu_memory_fraction", config.free_gpu_memory_fraction),
        ] {
            if let Some(value) = value {
                object.insert(name.to_owned(), serde_json::json!(value));
            }
        }
        object.insert(
            "cuda_graph_reserved_bytes".to_owned(),
            serde_json::json!(config.cuda_graph_reserved_bytes),
        );
    }
    role.rank.timing_model = TimingModelConfig::External {
        provider,
        config: resolved,
    };
    Ok(Some(Arc::new(timing)))
}

fn runtime_paths(traffic: &RuntimeTraffic) -> Result<Vec<PathBuf>> {
    let mut paths = traffic
        .trace_paths
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty()
        && let Some(path) = traffic.trace_path.as_deref()
    {
        paths.push(PathBuf::from(path));
    }
    ensure!(
        !paths.is_empty(),
        "trace traffic requires at least one path"
    );
    Ok(paths)
}

/// The largest trace block that divides both the cached prefix and the
/// engine's scheduler block (the engine block when there is no prefix).
/// Requests then share exactly the engine blocks inside the prefix, as with
/// one-token trace blocks, without one hash ID per token. Grouped prefixes are
/// sized in whole trace blocks, so they keep one-token blocks with a cached
/// prefix. The driver still hashes prompts at `engine_block_size`.
fn synthetic_trace_block_size(
    cached_prefix_tokens: usize,
    shared_prefix_ratio: f64,
    engine_block_size: usize,
) -> usize {
    if cached_prefix_tokens > 0 && shared_prefix_ratio > 0.0 {
        return 1;
    }
    let (mut a, mut b) = (engine_block_size, cached_prefix_tokens);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn synthetic_arrivals(traffic: &RuntimeTraffic) -> Result<ArrivalSpec> {
    match traffic.load_type.as_deref().unwrap_or("concurrency") {
        "concurrency" | "kv_capacity_fraction" => Ok(ArrivalSpec::Burst),
        "poisson" => {
            let qps = traffic
                .request_rate
                .context("poisson traffic requires request_rate")?;
            ensure!(
                qps.is_finite() && qps > 0.0,
                "request_rate must be positive"
            );
            Ok(ArrivalSpec::PoissonQps { qps })
        }
        "constant_rate" => {
            let qps = if let Some(qps) = traffic.request_rate {
                qps
            } else {
                let interval = traffic
                    .arrival_interval_ms
                    .context("constant_rate traffic requires an interval or rate")?;
                ensure!(
                    interval.is_finite() && interval > 0.0,
                    "arrival_interval_ms must be positive"
                );
                1_000.0 / interval
            };
            ensure!(
                qps.is_finite() && qps > 0.0,
                "request_rate must be positive"
            );
            Ok(ArrivalSpec::ConstantQps { qps })
        }
        other => Err(anyhow!("unsupported synthetic load type {other:?}")),
    }
}

fn concrete_session_count(traffic: &RuntimeTraffic) -> Result<usize> {
    if let Some(count) = traffic.request_count {
        ensure!(count > 0, "request_count must be positive");
        return Ok(count);
    }
    let ratio = traffic
        .num_request_ratio
        .context("synthetic traffic requires request_count or num_request_ratio")?;
    ensure!(
        ratio.is_finite() && ratio > 0.0,
        "num_request_ratio must be positive"
    );
    let load = traffic
        .concurrency
        .map(|value| value as f64)
        .or(traffic.request_rate)
        .or_else(|| {
            traffic
                .arrival_interval_ms
                .filter(|value| *value > 0.0)
                .map(|value| 1_000.0 / value)
        })
        .context("relative synthetic stop requires a concrete load")?;
    Ok(((ratio * load).round() as usize).max(1))
}

/// Per-request image workload declared by the Python traffic payload, if any.
fn synthetic_images(traffic: &RuntimeTraffic) -> Result<Option<SyntheticImages>> {
    let Some(count) = traffic.image_count else {
        return Ok(None);
    };
    Ok(Some(SyntheticImages {
        count,
        visual_tokens: traffic
            .image_visual_tokens
            .context("image traffic requires image_visual_tokens")?,
        encoder: crate::engine::EncoderShape {
            sequences: traffic
                .image_sequences
                .context("image traffic requires image_sequences")?,
            patch_tokens: traffic
                .image_patch_tokens
                .context("image traffic requires image_patch_tokens")?,
            transformer_tokens: traffic
                .image_transformer_tokens
                .context("image traffic requires image_transformer_tokens")?,
            output_tokens: traffic
                .image_output_tokens
                .context("image traffic requires image_output_tokens")?,
        },
        embedding_bytes: traffic
            .image_embedding_bytes
            .context("image traffic requires image_embedding_bytes")?,
        identity_pool: traffic.image_identity_pool,
    }))
}

fn resolve_kv_capacity_concurrency(
    traffic: &mut RuntimeTraffic,
    role: &ReplayRoleConfig,
    replicas: usize,
) -> Result<Option<usize>> {
    if traffic.load_type.as_deref() != Some("kv_capacity_fraction") {
        return Ok(None);
    }
    let ratio = traffic
        .kv_load_ratio
        .as_ref()
        .and_then(serde_json::Value::as_f64)
        .context("kv_capacity_fraction requires one concrete ratio")?;
    ensure!(
        ratio.is_finite() && ratio > 0.0,
        "KV load ratio must be positive"
    );
    let isl = traffic.isl.context("KV load requires isl")?;
    let osl = traffic.osl.context("KV load requires osl")?;
    let visual_tokens = synthetic_images(traffic)?.map_or(Some(0), |images| {
        images.count.checked_mul(images.visual_tokens)
    });
    let expected_tokens = visual_tokens
        .and_then(|visual_tokens| isl.checked_add(visual_tokens))
        .and_then(|tokens| tokens.checked_add(osl / 2))
        .context("KV-load expected token count overflow")?;
    ensure!(
        expected_tokens > 0,
        "KV load requires positive token lengths"
    );
    let capacity = if !role.rank.kv_cache_groups.is_empty() {
        let request_bytes = role
            .rank
            .kv_cache_groups
            .iter()
            .try_fold(0u64, |sum, group| {
                sum.checked_add(group.peak_request_bytes(expected_tokens as u64, 1)?)
                    .context("grouped KV request bytes overflow")
            })?;
        let per_rank = role
            .rank
            .kv_cache_capacity_bytes
            .context("grouped KV load requires byte capacity")?
            / request_bytes;
        usize::try_from(per_rank)?
            .checked_mul(role.dp_size as usize)
            .and_then(|value| value.checked_mul(replicas))
            .context("aggregate grouped KV concurrency overflow")?
    } else {
        let per_rank_tokens = role
            .rank
            .num_gpu_blocks
            .checked_mul(role.rank.block_size)
            .context("KV capacity overflow")?;
        let total_tokens = per_rank_tokens
            .checked_mul(role.dp_size as usize)
            .and_then(|value| value.checked_mul(replicas))
            .context("aggregate KV capacity overflow")?;
        total_tokens / expected_tokens
    };
    ensure!(
        capacity > 0,
        "candidate KV capacity cannot hold one request"
    );
    let concurrency = ((ratio * capacity as f64) as usize).max(1);
    traffic.concurrency = Some(concurrency);
    Ok(Some(concurrency))
}

fn build_agentic_driver(
    graph: ValidatedAgenticGraph,
    traffic: &RuntimeTraffic,
    engine_block_size: usize,
    speedup: f64,
) -> Result<WorkloadDriver> {
    if let Some(options) = &traffic.agentic_snapshot {
        let lanes = traffic
            .agentic_lanes
            .context("agentic_snapshot requires positive agentic_lanes")?;
        // Sample recorded time before applying speedup to remaining timers.
        let prepared = graph.into_prepared_snapshots(lanes, options.clone())?;
        let mut driver = if traffic.agentic_warmup {
            WorkloadDriver::new_agentic_warmup_with_policy(
                prepared,
                engine_block_size,
                true,
                speedup,
                traffic.agentic_warmup_policy,
                traffic
                    .agentic_profile
                    .as_ref()
                    .map_or(10_000.0, |profile| profile.global_idle_cap_seconds * 1000.0),
            )
        } else {
            WorkloadDriver::new_agentic_snapshots(prepared, engine_block_size, true, speedup)
        }?;
        if let Some(profile) = &traffic.agentic_profile {
            driver.enable_agentic_profile(profile.clone())?;
        }
        Ok(driver)
    } else {
        WorkloadDriver::new_agentic_trace_with_options(
            graph.normalize_starts().speed_up_timing(speedup)?,
            engine_block_size,
            true,
            traffic.agentic_lanes,
        )
    }
}

fn build_runtime_input(
    traffic: RuntimeTraffic,
    engine_block_size: usize,
) -> Result<BuiltRuntimeInput> {
    ensure!(engine_block_size > 0, "engine block size must be positive");
    ensure!(
        traffic.agentic_warmup_policy != AgenticWarmupPolicy::LiveTrajectoryV1
            || (traffic.agentic_warmup && traffic.agentic_profile.is_some()),
        "live_trajectory_v1 requires agentic_warmup and agentic_profile"
    );
    if let Some(profile) = &traffic.agentic_profile {
        profile.validate()?;
        ensure!(
            traffic.agentic_snapshot.is_some(),
            "agentic_profile requires agentic_snapshot"
        );
        ensure!(
            traffic.max_sim_time_ms.is_none(),
            "agentic_profile cannot be combined with max virtual time"
        );
    }
    ensure!(
        !traffic.agentic_warmup || traffic.agentic_snapshot.is_some(),
        "agentic_warmup requires agentic_snapshot"
    );
    if traffic.agentic_snapshot.is_some() {
        ensure!(
            traffic.source_type == "trace"
                && traffic.load_type.as_deref() == Some("trace_timestamps")
                && traffic.agentic_lanes.is_some_and(|lanes| lanes > 0)
                && traffic.replay_concurrency.is_none(),
            "agentic_snapshot requires trace_timestamps agentic input with positive agentic_lanes"
        );
    }
    ensure!(
        traffic.source_type == "trace" || traffic.agentic_lanes.is_none(),
        "agentic_lanes requires agentic trace input"
    );
    ensure!(
        traffic.source_type == "trace" || traffic.weka_nested_timestamp_basis.is_none(),
        "weka_nested_timestamp_basis requires Weka trace input"
    );
    if traffic.source_type == "trace" {
        let paths = runtime_paths(&traffic)?;
        let trace_block_size = traffic.trace_block_size.unwrap_or(512);
        let format = traffic.trace_format.as_deref().unwrap_or("mooncake");
        let speedup = traffic.arrival_speedup_ratio.unwrap_or(1.0);
        ensure!(
            format == "weka" || traffic.weka_nested_timestamp_basis.is_none(),
            "weka_nested_timestamp_basis requires Weka input"
        );
        ensure!(
            traffic.agentic_lanes != Some(0),
            "agentic_lanes must be greater than 0"
        );
        if traffic.agentic_lanes.is_some() {
            ensure!(
                matches!(format, "weka" | "agentic_mooncake" | "dynamo"),
                "agentic_lanes requires weka, agentic_mooncake, or agentic Dynamo input"
            );
        }
        if format == "agentic_mooncake" {
            require_agentic_execution_model(&traffic)?;
            ensure!(
                traffic.load_type.as_deref() == Some("trace_timestamps"),
                "agentic_mooncake requires trace_timestamps load"
            );
            ensure!(
                traffic.max_sim_time_ms.is_none(),
                "agentic trace does not support max virtual time"
            );
            ensure!(
                paths.len() == 1,
                "agentic_mooncake requires exactly one path"
            );
            ensure!(
                traffic.replay_concurrency.is_none(),
                "agentic_mooncake does not support concurrency load"
            );
            let trace = load_agentic_mooncake(&paths[0], trace_block_size)?;
            return Ok(BuiltRuntimeInput::without_weka_basis(
                ReplayRuntimeInput::Workload(build_agentic_driver(
                    trace,
                    &traffic,
                    engine_block_size,
                    speedup,
                )?),
            ));
        }
        if format == "weka" {
            require_agentic_execution_model(&traffic)?;
            ensure!(
                traffic.load_type.as_deref() == Some("trace_timestamps"),
                "weka requires trace_timestamps load"
            );
            ensure!(
                traffic.max_sim_time_ms.is_none(),
                "Weka agentic trace does not support max virtual time"
            );
            ensure!(paths.len() == 1, "weka requires exactly one path");
            ensure!(
                traffic.replay_concurrency.is_none(),
                "Weka agentic trace does not support concurrency load"
            );
            let requested_basis = traffic.weka_nested_timestamp_basis.unwrap_or_default();
            let (trace, resolved_basis, _cache_hit) = load_weka_agentic_graph_cached(
                &paths[0],
                traffic.trace_block_size,
                WekaImportOptions {
                    nested_timestamp_basis: requested_basis,
                },
                traffic.trace_graph_cache,
            )?;
            return Ok(BuiltRuntimeInput {
                input: ReplayRuntimeInput::Workload(build_agentic_driver(
                    trace,
                    &traffic,
                    engine_block_size,
                    speedup,
                )?),
                weka_nested_timestamp_basis: Some(resolved_basis),
            });
        }
        if format == "dynamo" {
            let loaded =
                DynamoRequestTrace::from_request_trace_files(&paths, traffic.trace_block_size)?;
            let driver = match loaded {
                DynamoRequestTrace::Standard(trace) => {
                    ensure!(
                        traffic.agentic_lanes.is_none(),
                        "agentic_lanes requires an agentic Dynamo trace"
                    );
                    let trace = trace.normalize_session_starts()?.speed_up_timing(speedup)?;
                    match traffic.replay_concurrency {
                        Some(cap) => {
                            WorkloadDriver::new_concurrency(trace, engine_block_size, cap)?
                        }
                        None => WorkloadDriver::new_trace(trace, engine_block_size)?,
                    }
                }
                DynamoRequestTrace::Agentic(trace) => {
                    require_agentic_execution_model(&traffic)?;
                    ensure!(
                        traffic.replay_concurrency.is_none(),
                        "agentic Dynamo trace does not support concurrency load"
                    );
                    ensure!(
                        traffic.max_sim_time_ms.is_none(),
                        "agentic Dynamo trace does not support max virtual time"
                    );
                    build_agentic_driver(trace, &traffic, engine_block_size, speedup)?
                }
            };
            return Ok(BuiltRuntimeInput::without_weka_basis(
                ReplayRuntimeInput::Workload(driver),
            ));
        }
        ensure!(
            paths.len() == 1,
            "trace format {format:?} requires exactly one path"
        );
        let mut trace = match format {
            "mooncake" | "mooncake-delta" => Trace::from_mooncake(&paths[0], trace_block_size)?,
            "applied_compute_agentic" => {
                Trace::from_applied_compute_agentic(&paths[0], trace_block_size, 0.0, 0)?
            }
            other => return Err(anyhow!("unsupported trace format {other:?}")),
        };
        trace = trace.normalize_session_starts()?.speed_up_timing(speedup)?;
        let delta = format == "mooncake-delta";
        let concurrency = traffic.replay_concurrency;
        let driver = match (concurrency, delta) {
            (Some(cap), true) => {
                WorkloadDriver::new_concurrency_accumulating_deltas(trace, engine_block_size, cap)?
            }
            (Some(cap), false) => WorkloadDriver::new_concurrency(trace, engine_block_size, cap)?,
            (None, true) => {
                WorkloadDriver::new_trace_accumulating_deltas(trace, engine_block_size)?
            }
            (None, false) => WorkloadDriver::new_trace(trace, engine_block_size)?,
        };
        return Ok(BuiltRuntimeInput::without_weka_basis(
            ReplayRuntimeInput::Workload(driver),
        ));
    }

    ensure!(
        matches!(
            traffic.source_type.as_str(),
            "synthetic" | "synthetic-session"
        ),
        "unsupported synthetic source type {:?}",
        traffic.source_type
    );
    let sessions = concrete_session_count(&traffic)?;
    let turns = if traffic.source_type == "synthetic-session" {
        traffic.turns_per_session.unwrap_or(4)
    } else {
        1
    };
    let cached_prefix_tokens = traffic.cached_prefix_tokens.unwrap_or(0);
    let images = synthetic_images(&traffic)?;
    let shared_prefix_ratio = traffic.shared_prefix_ratio.unwrap_or(0.0);
    let trace = Trace::synthetic(SyntheticTraceSpec {
        // Image placeholder spans stay token-exact on a one-token trace block; the
        // driver still hashes them at `engine_block_size` when it constructs
        // replay requests.
        block_size: if images.is_some() {
            1
        } else {
            synthetic_trace_block_size(cached_prefix_tokens, shared_prefix_ratio, engine_block_size)
        },
        num_sessions: sessions,
        turns_per_session: turns,
        input_tokens: LengthSpec {
            mean: traffic.isl.context("synthetic traffic requires isl")?,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: traffic.osl.context("synthetic traffic requires osl")?,
            stddev: 0.0,
        },
        cached_prefix_tokens,
        shared_prefix_ratio,
        num_prefix_groups: traffic.num_prefix_groups.unwrap_or(0),
        first_turn_arrivals: synthetic_arrivals(&traffic)?,
        inter_turn_delays: traffic
            .inter_turn_delay_ms
            .filter(|delay| *delay > 0.0)
            .map_or(DelaySpec::None, DelaySpec::ConstantMs),
        seed: 0,
        arrival_seed: traffic.arrival_seed.unwrap_or(42),
        images,
    })?;
    let cap = traffic.concurrency;
    let accumulate = traffic.source_type == "synthetic-session";
    let driver = match (cap, accumulate) {
        (Some(cap), true) => {
            WorkloadDriver::new_concurrency_accumulating_deltas(trace, engine_block_size, cap)?
        }
        (Some(cap), false) => WorkloadDriver::new_concurrency(trace, engine_block_size, cap)?,
        (None, true) => WorkloadDriver::new_trace_accumulating_deltas(trace, engine_block_size)?,
        (None, false) => WorkloadDriver::new_trace(trace, engine_block_size)?,
    };
    Ok(BuiltRuntimeInput::without_weka_basis(
        ReplayRuntimeInput::Workload(driver),
    ))
}

fn run_with_input(
    spec: ReplaySpec,
    factory: ReplayEngineFactory,
    input: Option<ReplayRuntimeInput>,
    capture_artifacts: bool,
    determinism: ReplayDeterminism,
    serving_path: ServingPath,
) -> crate::replay::ReplayResult<(crate::replay::ReplayReport, Option<ReplayArtifacts>)> {
    let mut replayer = match input {
        Some(input) => Replayer::new(spec, factory)?.with_runtime_input(input),
        None => Replayer::new(spec, factory)?,
    }
    .with_capture_options(ReplayCaptureOptions {
        determinism,
        ..ReplayCaptureOptions::default()
    });
    if let Some(tokenization) = serving_path.tokenization {
        replayer = replayer.with_tokenization(
            tokenization.workers,
            tokenization.base_ms,
            tokenization.per_input_token_ms,
        )?;
    }
    if let Some(detokenization) = serving_path.detokenization {
        replayer = replayer.with_detokenization(
            detokenization.workers,
            detokenization.base_ms,
            detokenization.per_output_ms,
        )?;
    }
    if capture_artifacts {
        let (report, artifacts) =
            replayer.run_with_artifacts(ReplayArtifactKvEventVisibility::Native)?;
        Ok((report, Some(artifacts)))
    } else {
        Ok((replayer.run()?, None))
    }
}

struct TimingPowerSource {
    timing: Arc<dyn TimingModel>,
    prefill_speedup_ratio: f64,
    decode_speedup_ratio: f64,
}

impl TimingPowerSource {
    fn new(timing: Arc<dyn TimingModel>, config: &EngineConfig) -> Self {
        Self {
            timing,
            prefill_speedup_ratio: config.speedup_ratio,
            decode_speedup_ratio: config.speedup_ratio * config.decode_speedup_ratio,
        }
    }
}

fn replay_timing_evidence(sources: &[TimingPowerSource]) -> Result<Option<TimingEvidenceSummary>> {
    replay_timing_evidence_with_fpm(sources, false)
}

fn replay_timing_evidence_with_fpm(
    sources: &[TimingPowerSource],
    include_fpm: bool,
) -> Result<Option<TimingEvidenceSummary>> {
    let mut combined = TimingEvidenceSummary::default();
    for source in sources {
        let Some(summary) = source.timing.evidence_summary().or_else(|| {
            include_fpm
                .then(|| source.timing.fpm_evidence_summary())
                .flatten()
        }) else {
            return Ok(None);
        };
        combined.prefill.try_accumulate(scale_power_phase(
            summary.prefill,
            source.prefill_speedup_ratio,
        )?)?;
        combined.decode.try_accumulate(scale_power_phase(
            summary.decode,
            source.decode_speedup_ratio,
        )?)?;
    }
    Ok(Some(combined))
}

fn replay_fpm_query_evidence(sources: &[TimingPowerSource]) -> Result<Option<serde_json::Value>> {
    let mut providers = Vec::new();
    for (provider_index, source) in sources.iter().enumerate() {
        let Some(summary) = source.timing.fpm_evidence_summary() else {
            continue;
        };
        let mut phases = Vec::new();
        for (name, phase, speedup) in [
            ("prefill", summary.prefill, source.prefill_speedup_ratio),
            ("decode", summary.decode, source.decode_speedup_ratio),
        ] {
            let phase = scale_power_phase(phase, speedup)?;
            let operations = phase
                .operations
                .into_iter()
                .filter_map(|operation| {
                    let details = operation.details?;
                    (!details.fpm_estimates.is_empty()).then(|| {
                        serde_json::json!({
                            "name": operation.name, "fpm_estimates": details.fpm_estimates,
                        })
                    })
                })
                .collect::<Vec<_>>();
            phases.push(serde_json::json!({"name": name, "operations": operations}));
        }
        providers.push(serde_json::json!({"provider_index": provider_index, "phases": phases}));
    }
    Ok((!providers.is_empty()).then(|| {
        serde_json::json!({
            "schema_version": "1.0", "aggregation": "identical_estimates_with_invocation_counts",
            "providers": providers,
        })
    }))
}

fn replay_power_stats(summary: &TimingEvidenceSummary) -> Result<TracePowerStats> {
    let mut combined = summary.prefill.clone();
    combined.try_accumulate(summary.decode.clone())?;
    phase_power_stats(&combined)
}

fn phase_power_stats(combined: &TimingPhaseEvidence) -> Result<TracePowerStats> {
    if combined.latency_ms <= 0.0 {
        return TracePowerStats::new(None, 0.0);
    }
    // Keep uncovered latency separate: subtracting two rounded totals loses
    // the exact 0.27 covered + 0.03 uncovered boundary. C >= 9U expresses the
    // same 90% gate without a tolerance that would admit the next lower case.
    let uncovered: f64 = if combined.operations.is_empty() {
        combined.latency_ms - combined.covered_latency_ms
    } else {
        combined
            .operations
            .iter()
            .map(|op| op.latency_ms - op.covered_latency_ms)
            .sum()
    };
    let qualifies =
        combined.covered_latency_ms > 0.0 && combined.covered_latency_ms / 9.0 >= uncovered;
    let ratio = (combined.covered_latency_ms / combined.latency_ms).clamp(0.0, 1.0);
    let coverage = if qualifies {
        ratio.max(POWER_DATA_COVERAGE_THRESHOLD)
    } else {
        ratio.min(POWER_DATA_COVERAGE_THRESHOLD.next_down())
    };
    let power_w = qualifies
        .then(|| {
            combined
                .energy_wms
                .map(|energy| energy / combined.latency_ms)
        })
        .flatten()
        .filter(|power| power.is_finite() && *power > 0.0);
    TracePowerStats::new(power_w, coverage)
}

fn replay_performance_diagnostics(summary: Option<&TimingEvidenceSummary>) -> serde_json::Value {
    let scope = "accumulated_active_forward_pass_per_gpu";
    let Some(summary) = summary else {
        return serde_json::json!({"status": "unavailable", "scope": scope, "latency_unit": "ms", "phases": [],
            "unavailable_reason": "selected timing provider does not export operation diagnostics"});
    };
    let phases = [("prefill", &summary.prefill), ("decode", &summary.decode)].into_iter().map(|(name, phase)| {
        let mut operations = phase.operations.iter().map(|op| {
            let sol = op.details.as_ref().and_then(|d| d.sol.as_ref());
            let mut record = serde_json::json!({"name": op.name, "latency_ms": op.latency_ms,
                "source": op.source.as_str(),
                "sol": sol,
                "sol_unavailable_reason": if sol.is_some() { None } else {
                    op.details.as_ref().and_then(|d| d.sol_unavailable_reason.as_deref()).or(Some("provider did not export SOL evidence"))
                },
                "latency_to_sol_ratio": sol.filter(|s| s.latency_ms > 0.0).map(|s| op.latency_ms / s.latency_ms).filter(|r| r.is_finite()),
                "fallbacks": op.details.as_ref().map(|d| &d.fallbacks),
            });
            if let Some(details) = &op.details && !details.fpm_estimates.is_empty() {
                record["fpm_estimates"] = serde_json::to_value(&details.fpm_estimates).expect("finite native FPM evidence");
            }
            record
        }).collect::<Vec<_>>();
        operations.sort_by(|a,b| a["name"].as_str().cmp(&b["name"].as_str()));
        let sol = phase.operations.iter().map(|op| op.details.as_ref()?.sol.as_ref()).collect::<Option<Vec<_>>>()
            .filter(|ops| !ops.is_empty()).and_then(|ops| {
                let latency_ms = ops.iter().map(|s| s.latency_ms).sum::<f64>();
                let math_ms = ops.iter().map(|s| s.math_ms).sum::<f64>();
                let memory_ms = ops.iter().map(|s| s.memory_ms).sum::<f64>();
                [latency_ms, math_ms, memory_ms].iter().all(|v| v.is_finite()).then(||
                    serde_json::json!({"latency_ms": latency_ms, "math_ms": math_ms, "memory_ms": memory_ms}))
            });
        serde_json::json!({"name": name, "latency_ms": phase.latency_ms, "sol": sol,
            "sol_unavailable_reason": if sol.is_some() { None } else { Some("one or more operations lack SOL evidence, or this phase was not observed") },
            "operations": operations})
    }).collect::<Vec<_>>();
    let observed = phases
        .iter()
        .any(|phase| !phase["operations"].as_array().unwrap().is_empty());
    serde_json::json!({"status": if observed { "available" } else { "not_observed" },
        "scope": scope, "latency_unit": "ms", "phases": phases})
}

fn replay_power_diagnostics(
    summary: Option<&TimingEvidenceSummary>,
    unavailable_reason: Option<&'static str>,
) -> Result<ReplayPowerDiagnostics> {
    let Some(summary) = summary else {
        return Ok(ReplayPowerDiagnostics {
            schema_version: "1.0",
            scope: "active_forward_pass_per_gpu",
            power_w_unit: "W",
            energy_unit: "W-ms",
            latency_unit: "ms",
            coverage_gate: POWER_DATA_COVERAGE_THRESHOLD,
            publication_status: "unsupported",
            energy_wms: None,
            latency_ms: None,
            covered_latency_ms: None,
            power_w: None,
            power_coverage: None,
            unavailable_reason,
            phases: Vec::new(),
        });
    };

    let stats = replay_power_stats(summary)?;
    let mut combined = summary.prefill.clone();
    combined.try_accumulate(summary.decode.clone())?;
    let observed = combined.latency_ms > 0.0;
    Ok(ReplayPowerDiagnostics {
        schema_version: "1.0",
        scope: "active_forward_pass_per_gpu",
        power_w_unit: "W",
        energy_unit: "W-ms",
        latency_unit: "ms",
        coverage_gate: POWER_DATA_COVERAGE_THRESHOLD,
        publication_status: publication_status(stats.power_w, stats.coverage, observed),
        energy_wms: combined.energy_wms.filter(|energy| *energy > 0.0),
        latency_ms: Some(combined.latency_ms),
        covered_latency_ms: Some(combined.covered_latency_ms),
        power_w: stats.power_w,
        power_coverage: Some(stats.coverage),
        unavailable_reason: (!observed).then_some("no forward-pass timing evidence was observed"),
        phases: vec![
            phase_power_diagnostics("prefill", &summary.prefill)?,
            phase_power_diagnostics("decode", &summary.decode)?,
        ],
    })
}

fn phase_power_diagnostics(
    name: &'static str,
    phase: &TimingPhaseEvidence,
) -> Result<ReplayPhasePowerDiagnostics> {
    let stats = phase_power_stats(phase)?;
    let coverage = stats.coverage;
    let power_w = stats.power_w;
    let phase_energy = phase.energy_wms.filter(|energy| *energy > 0.0);
    let mut operations = phase
        .operations
        .iter()
        .map(|operation| operation_power_diagnostics(operation, phase_energy))
        .collect::<Vec<_>>();
    operations.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.source.cmp(&right.source))
    });
    let source = phase.source.as_ref().map_or_else(
        || "missing".to_string(),
        |source| source.as_str().to_string(),
    );
    Ok(ReplayPhasePowerDiagnostics {
        name,
        energy_wms: phase_energy,
        latency_ms: phase.latency_ms,
        covered_latency_ms: phase.covered_latency_ms,
        power_coverage: coverage,
        publication_status: publication_status(power_w, coverage, phase.latency_ms > 0.0),
        power_w,
        source_kind: evidence_source_kind(phase.source.as_ref(), phase_energy.is_some()),
        source,
        operations,
    })
}

fn operation_power_diagnostics(
    operation: &TimingOperationEvidence,
    phase_energy: Option<f64>,
) -> ReplayOperationPowerDiagnostics {
    let energy_wms = operation.energy_wms.filter(|energy| *energy > 0.0);
    let power_coverage = if operation.latency_ms > 0.0 {
        (operation.covered_latency_ms / operation.latency_ms).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (status, uncovered_reason) = if energy_wms.is_none() {
        (
            "missing",
            Some("timing provider returned latency without positive energy evidence"),
        )
    } else if operation.latency_ms == 0.0 {
        ("available", None)
    } else if power_coverage < 1.0 {
        (
            "partial",
            Some("some accumulated operation latency lacks positive energy evidence"),
        )
    } else {
        ("available", None)
    };
    ReplayOperationPowerDiagnostics {
        name: operation.name.clone(),
        energy_wms,
        latency_ms: operation.latency_ms,
        covered_latency_ms: operation.covered_latency_ms,
        power_coverage,
        energy_contribution: energy_wms
            .zip(phase_energy)
            .map(|(energy, total)| energy / total),
        source: operation.source.as_str().to_string(),
        source_kind: evidence_source_kind(Some(&operation.source), energy_wms.is_some()),
        status,
        uncovered_reason,
    }
}

fn publication_status(power_w: Option<f64>, coverage: f64, observed: bool) -> &'static str {
    if !observed {
        "not_observed"
    } else if power_w.is_some() {
        "available"
    } else if coverage < POWER_DATA_COVERAGE_THRESHOLD {
        "withheld"
    } else {
        "missing"
    }
}

fn evidence_source_kind(
    source: Option<&TimingEvidenceSource>,
    energy_available: bool,
) -> &'static str {
    if !energy_available {
        return "missing";
    }
    match source.map(TimingEvidenceSource::as_str) {
        Some("silicon" | "empirical") => "measured",
        Some("transferred") => "transferred",
        Some("sol" | "estimated") => "modeled",
        Some("mixed") => "mixed",
        Some(_) => "other",
        None => "missing",
    }
}

fn scale_power_phase(
    mut phase: TimingPhaseEvidence,
    speedup_ratio: f64,
) -> Result<TimingPhaseEvidence> {
    ensure!(
        speedup_ratio.is_finite() && speedup_ratio >= 0.0,
        "modeled speedup ratio must be finite and non-negative, got {speedup_ratio}"
    );
    let scale = if speedup_ratio > 0.0 {
        speedup_ratio.recip()
    } else {
        1.0
    };
    phase.energy_wms = phase.energy_wms.map(|energy| energy * scale);
    phase.latency_ms *= scale;
    phase.covered_latency_ms *= scale;
    for operation in &mut phase.operations {
        operation.energy_wms = operation.energy_wms.map(|energy| energy * scale);
        operation.latency_ms *= scale;
        operation.covered_latency_ms *= scale;
        if let Some(details) = &mut operation.details {
            for evidence in &mut details.fpm_estimates {
                evidence.latency_scale *= scale;
            }
        }
        // SOL compares the same scheduled work, before synthetic speedup; it
        // remains an unscaled physical baseline.
    }
    Ok(phase)
}

fn infer_aic_timing_topology(engine: &mut serde_json::Value) -> Result<()> {
    let timing = engine.get("rank").and_then(|rank| rank.get("timing_model"));
    if let Some(timing) = timing
        && timing.get("type").and_then(serde_json::Value::as_str) == Some("external")
        && timing.get("provider").and_then(serde_json::Value::as_str) == Some("aic")
    {
        let config = timing.get("config").cloned().unwrap_or_default();
        for (target, fields) in [
            ("tensor_parallel_size", ["tp", "tp_size"]),
            ("dp_size", ["attention_dp", "attention_dp_size"]),
            ("prefill_context_parallel_size", ["cp_size", "cp_size"]),
        ] {
            // An unknown size leaves the engine's own topology.
            if let Some(value) = config
                .get(fields[0])
                .or_else(|| config.get(fields[1]))
                .filter(|value| !value.is_null() && value.as_str() != Some(crate::UNKNOWN_IDENTITY))
            {
                if let Some(explicit) = engine.get(target) {
                    ensure!(
                        explicit == value,
                        "{target} conflicts with canonical AIC timing topology"
                    );
                } else if let Some(object) = engine.as_object_mut() {
                    object.insert(target.to_owned(), value.clone());
                }
            }
        }
    }
    for role in ["prefill", "decode"] {
        if let Some(child) = engine.get_mut(role) {
            infer_aic_timing_topology(child)?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ReplayCoverageFailure {
    error: anyhow::Error,
    coverage: serde_json::Value,
}

impl std::fmt::Display for ReplayCoverageFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.error)
    }
}

impl std::error::Error for ReplayCoverageFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.error.source()
    }
}

fn replay_failure(
    error: anyhow::Error,
    coverage: Result<Option<serde_json::Value>>,
) -> anyhow::Error {
    match coverage {
        Ok(Some(coverage)) => ReplayCoverageFailure { error, coverage }.into(),
        Ok(None) => error,
        Err(coverage_error) => error.context(format!(
            "FPM coverage diagnostics failed: {coverage_error:#}"
        )),
    }
}

fn replay_fpm_coverage(
    sources: &[(ForwardPassWorkerType, Arc<ForwardPassPerfModel>)],
    report: Option<&crate::replay::ReplayReport>,
) -> Result<Option<serde_json::Value>> {
    if sources.is_empty() {
        return Ok(None);
    }
    let mut measured = 0u64;
    let mut interpolated = 0u64;
    let mut unsupported = 0u64;
    let mut roles = Vec::with_capacity(sources.len());
    for (role, model) in sources {
        let coverage = model
            .fpm_query_coverage()?
            .context("requested FPM query coverage is unavailable")?;
        measured += coverage.queries.measured;
        interpolated += coverage.queries.interpolated;
        unsupported += coverage.queries.unsupported;
        roles.push(serde_json::json!({"worker_type": role, "coverage": coverage}));
    }
    let replay_completed = report.is_some_and(|report| {
        let counts = &report.request_counts;
        counts.num_requests > 0
            && counts.completed_requests == counts.num_requests
            && report.agentic_graph.as_ref().is_none_or(|graph| {
                graph.node_count == counts.completed_requests
                    && report
                        .agentic_play_outcomes
                        .as_ref()
                        .is_some_and(|outcomes| {
                            outcomes.len() == graph.play_count
                                && outcomes.iter().all(|outcome| {
                                    matches!(
                                        outcome.status,
                                        crate::replay::loadgen::AgenticPlayStatus::Completed
                                    )
                                })
                        })
            })
    });
    let covered = replay_completed && measured + interpolated > 0 && unsupported == 0;
    Ok(Some(serde_json::json!({
        "schema_version": 1,
        "status": if covered { "covered" } else { "incomplete" },
        "replay_completed": replay_completed,
        "counting_unit": "native_lookup_resolutions",
        "queries": {"measured": measured, "interpolated": interpolated, "unsupported": unsupported},
        "roles": roles,
        "accuracy": "not_assessed",
    })))
}

fn execute_json(payload: &str, capture_artifacts: bool) -> Result<String> {
    let (mut spec, mut traffic, capture_performance_diagnostics, determinism, serving_path) =
        match serde_json::from_str(payload).context("invalid AISimulate execution ReplaySpec")? {
            ExecutionPayload::Configured {
                spec,
                traffic,
                capture_performance_diagnostics,
                determinism,
                tokenization,
                detokenization,
            } => (
                spec,
                traffic.map(|t| *t),
                capture_performance_diagnostics,
                determinism,
                ServingPath {
                    tokenization,
                    detokenization,
                },
            ),
            ExecutionPayload::Legacy(spec) => (
                spec,
                None,
                false,
                ReplayDeterminism::default(),
                ServingPath::default(),
            ),
        };
    ensure!(
        serving_path.detokenization.is_none()
            || traffic
                .as_ref()
                .is_none_or(|traffic| traffic.agentic_profile.is_none()),
        "detokenization cannot be combined with agentic_profile"
    );
    ensure!(
        spec.max_sim_time_ms.is_none()
            || traffic
                .as_ref()
                .is_none_or(|traffic| traffic.agentic_profile.is_none()),
        "agentic_profile cannot be combined with ReplaySpec.max_sim_time_ms"
    );
    // The pool carries every request's images; the language ranks run no vision tower.
    ensure!(
        spec.encoder.is_none()
            || traffic
                .as_ref()
                .is_none_or(|traffic| traffic.image_count.is_none()),
        "image traffic cannot be combined with an encoder pool"
    );
    let agentic_input = traffic.as_ref().and_then(|traffic| {
        traffic
            .trace_format
            .as_deref()
            .filter(|format| matches!(*format, "weka" | "agentic_mooncake" | "dynamo"))
            .map(|format| {
                (
                    format.to_string(),
                    traffic.agentic_lanes,
                    traffic.execution_model.clone(),
                )
            })
    });
    if capture_artifacts {
        ensure!(
            matches!(&spec.topology, ReplayTopology::Aggregated { .. }),
            "detailed replay artifacts require aggregated topology"
        );
    }
    let serialized_engine = spec.engine.clone();
    let record_per_request = spec.record_per_request;
    infer_aic_timing_topology(&mut spec.engine)?;
    let mut engine_config: ReplayEngineConfig = if spec.engine.is_null() {
        ReplayEngineConfig::default()
    } else {
        serde_json::from_value(spec.engine.clone())
            .context("invalid native engine descriptor in execution ReplaySpec")?
    };
    let mut coverage_sources = Vec::new();
    let encoder_timing = spec
        .encoder
        .as_ref()
        .map(|encoder| {
            resolve_encoder_timing(
                encoder,
                capture_performance_diagnostics,
                &mut coverage_sources,
            )
        })
        .transpose()?
        .flatten();
    let expected_power_sources = match &spec.topology {
        ReplayTopology::Aggregated { .. } => 1,
        ReplayTopology::Disaggregated { .. } => 2,
    } + usize::from(encoder_timing.is_some());
    let mut power_sources = Vec::with_capacity(expected_power_sources);
    if let Some(timing) = &encoder_timing {
        // The encoder pool's forwards join the power evidence like a rank's.
        power_sources.push(TimingPowerSource {
            timing: Arc::clone(timing),
            prefill_speedup_ratio: 1.0,
            decode_speedup_ratio: 1.0,
        });
    }
    let mut prediction_provenance = serde_json::Map::new();

    let replay_result = (|| -> Result<_> {
        match spec.topology.clone() {
        ReplayTopology::Aggregated { .. } => {
            let mut role = aggregated_role(&engine_config);
            let capacity_is_explicit = role
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, None));
            let timing = resolve_role_timing(
                &mut role,
                capacity_is_explicit,
                ForwardPassWorkerType::Aggregated,
                capture_performance_diagnostics,
                &mut coverage_sources,
            )?;
            if let Some(timing) = timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &role.rank));
            }
            prediction_provenance.insert("aggregated".into(), timing_provenance(&role.rank.timing_model));
            engine_config.dp_size = role.dp_size;
            engine_config.tensor_parallel_size = role.tensor_parallel_size;
            engine_config.prefill_context_parallel_size = role.prefill_context_parallel_size;
            engine_config.num_gpu_blocks_is_explicit = role.num_gpu_blocks_is_explicit;
            engine_config.rank = role.rank;
            if let Some(traffic) = traffic.as_mut()
                && let ReplayTopology::Aggregated { workers } = &spec.topology
                && let Some(concurrency) = resolve_kv_capacity_concurrency(
                    traffic,
                    &ReplayRoleConfig {
                        dp_size: engine_config.dp_size,
                        tensor_parallel_size: engine_config.tensor_parallel_size,
                        prefill_context_parallel_size: engine_config.prefill_context_parallel_size,
                        num_gpu_blocks_is_explicit: engine_config.num_gpu_blocks_is_explicit,
                        rank: engine_config.rank.clone(),
                    },
                    workers.initial_workers,
                )?
            {
                spec.max_in_flight = Some(concurrency);
            }
            spec.engine = serde_json::to_value(&engine_config)
                .context("serializing materialized native engine descriptor")?;
            let built_input = traffic
                .map(|traffic| build_runtime_input(traffic, engine_config.rank.block_size))
                .transpose()?;
            if let Some(built) = &built_input {
                validate_public_agentic_engine(&built.input, &spec, &engine_config)?;
            }
            let resolved_basis = built_input
                .as_ref()
                .and_then(|built| built.weka_nested_timestamp_basis);
            let input = built_input.map(|built| built.input);
            let mut factory = timing.map_or_else(
                ReplayEngineFactory::new,
                ReplayEngineFactory::with_timing_model,
            );
            if let Some(timing) = &encoder_timing {
                factory = factory.with_encoder_timing(Arc::clone(timing));
            }
            run_with_input(spec, factory, input, capture_artifacts, determinism, serving_path)
                .map(|(report, artifacts)| (report, artifacts, resolved_basis))
        }
        ReplayTopology::Disaggregated { .. } => {
            let mut prefill = engine_config
                .prefill
                .clone()
                .unwrap_or_else(|| aggregated_role(&engine_config));
            let mut decode = engine_config
                .decode
                .clone()
                .unwrap_or_else(|| aggregated_role(&engine_config));
            // Determine the loaded kind before compiling either timing model:
            // Dynamo inputs may contain ordinary or agentic requests. Retain
            // the built input so this validation does not load the trace twice.
            let trace_input = if traffic.as_ref().is_some_and(|traffic| {
                traffic.source_type == "trace"
                    && matches!(
                        traffic.trace_format.as_deref(),
                        Some("weka" | "agentic_mooncake" | "dynamo")
                    )
            }) {
                let traffic = traffic.as_ref().expect("trace traffic was checked");
                let built = build_runtime_input(traffic.clone(), prefill.rank.block_size)?;
                if let ReplayRuntimeInput::Workload(driver) = &built.input
                    && driver.is_agentic()
                {
                    let target = traffic.execution_model
                        .as_deref()
                        .map(str::trim)
                        .context("agentic execution requires a configured target model")?;
                    for (name, role) in [("prefill", &prefill), ("decode", &decode)] {
                        if let TimingModelConfig::External { provider, config } =
                            &role.rank.timing_model
                            && provider == "aic"
                        {
                            let model = config.get("model").and_then(serde_json::Value::as_str);
                            ensure!(
                                model == Some(target),
                                "{name} AIC timing model {model:?} must match agentic execution model {target:?}"
                            );
                        }
                    }
                }
                Some(built)
            } else {
                None
            };
            let prefill_capacity_is_explicit = prefill
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, Some("prefill")));
            let decode_capacity_is_explicit = decode
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, Some("decode")));
            let prefill_timing = resolve_role_timing(
                &mut prefill,
                prefill_capacity_is_explicit,
                ForwardPassWorkerType::Prefill,
                capture_performance_diagnostics,
                &mut coverage_sources,
            )?;
            let decode_timing = resolve_role_timing(
                &mut decode,
                decode_capacity_is_explicit,
                ForwardPassWorkerType::Decode,
                capture_performance_diagnostics,
                &mut coverage_sources,
            )?;
            if let Some(timing) = prefill_timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &prefill.rank));
            }
            if let Some(timing) = decode_timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &decode.rank));
            }
            prediction_provenance.insert("prefill".into(), timing_provenance(&prefill.rank.timing_model));
            prediction_provenance.insert("decode".into(), timing_provenance(&decode.rank.timing_model));
            engine_config.prefill = Some(prefill);
            engine_config.decode = Some(decode);
            if let Some(traffic) = traffic.as_mut()
                && let ReplayTopology::Disaggregated { decode, .. } = &spec.topology
                && let Some(concurrency) = resolve_kv_capacity_concurrency(
                    traffic,
                    engine_config
                        .decode
                        .as_ref()
                        .expect("decode role was materialized"),
                    decode.initial_workers,
                )?
            {
                spec.max_in_flight = Some(concurrency);
            }
            spec.engine = serde_json::to_value(&engine_config)
                .context("serializing materialized native engine descriptor")?;
            let built_input = match trace_input {
                Some(built) => Some(built),
                None => traffic.map(|traffic| {
                    build_runtime_input(
                        traffic,
                        engine_config
                            .prefill
                            .as_ref()
                            .expect("prefill role was materialized")
                            .rank
                            .block_size,
                    )
                })
                .transpose()?,
            };
            if let Some(built) = &built_input {
                validate_public_agentic_engine(&built.input, &spec, &engine_config)?;
            }
            let resolved_basis = built_input
                .as_ref()
                .and_then(|built| built.weka_nested_timestamp_basis);
            let input = built_input.map(|built| built.input);
            let mut factory =
                ReplayEngineFactory::with_optional_role_timing_models(prefill_timing, decode_timing);
            if let Some(timing) = &encoder_timing {
                factory = factory.with_encoder_timing(Arc::clone(timing));
            }
            run_with_input(spec, factory, input, capture_artifacts, determinism, serving_path)
            .map(|(report, artifacts)| (report, artifacts, resolved_basis))
        }
        }
        .map_err(Into::into)
    })()
    .context("AISimulate replay failed");
    let (mut report, artifacts, resolved_weka_timestamp_basis) = match replay_result {
        Ok(result) => result,
        Err(error) => {
            return Err(replay_failure(
                error,
                replay_fpm_coverage(&coverage_sources, None),
            ));
        }
    };
    let (timing_evidence, unavailable_reason) = if power_sources.len() == expected_power_sources {
        (
            replay_timing_evidence(&power_sources)?,
            Some(concat!(
                "timing provider does not expose typed operation energy evidence; ",
                "whole-model FPM and latency-only providers are unsupported"
            )),
        )
    } else {
        (
            None,
            Some("one or more replay roles have no timing provider"),
        )
    };
    if let Some(summary) = timing_evidence.as_ref() {
        report = report.with_power(Some(replay_power_stats(summary)?));
    }
    report = report.with_power_diagnostics(Some(replay_power_diagnostics(
        timing_evidence.as_ref(),
        unavailable_reason,
    )?));
    let mut report_json =
        serde_json::to_value(&report).context("serializing AISimulate replay report summary")?;
    report_json["prediction_provenance"] = prediction_provenance.into();
    if let Some(coverage) = replay_fpm_coverage(&coverage_sources, Some(&report))? {
        report_json["fpm_query_coverage"] = coverage;
    }
    if capture_performance_diagnostics {
        let performance_evidence = replay_timing_evidence_with_fpm(&power_sources, true)?;
        report_json["performance_diagnostics"] =
            replay_performance_diagnostics(performance_evidence.as_ref());
    }
    if let Some(evidence) = replay_fpm_query_evidence(&power_sources)? {
        report_json["fpm_query_evidence"] = evidence;
    }
    if report.agentic_graph.is_some()
        && let Some((input_format, agentic_lanes, execution_model)) = agentic_input
    {
        let source_models = report
            .agentic_graph
            .as_ref()
            .expect("agentic graph presence was checked")
            .source_models
            .clone();
        let execution_model = execution_model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .context("agentic execution did not declare its configured target model")?;
        let object = report_json
            .as_object_mut()
            .context("AISimulate replay report did not serialize as an object")?;
        object.insert(
            "agentic_qualification".to_string(),
            serde_json::Value::String("functional_only".to_string()),
        );
        object.insert(
            "agentic_input_format".to_string(),
            serde_json::Value::String(input_format),
        );
        object.insert(
            "agentic_lanes".to_string(),
            serde_json::to_value(agentic_lanes)
                .context("serializing configured agentic lane count")?,
        );
        if let Some(resolved_basis) = resolved_weka_timestamp_basis {
            object.insert(
                "weka_nested_timestamp_basis".to_string(),
                serde_json::Value::String(resolved_basis.as_str().to_string()),
            );
        }
        object.insert(
            "agentic_model_projection".to_string(),
            serde_json::json!({
                "policy": AGENTIC_MODEL_PROJECTION_POLICY,
                "source_models": source_models,
                "target_model": execution_model,
            }),
        );
    }
    if record_per_request || !report.per_request.is_empty() {
        // Holds the key's place in the map; the records are written there during
        // serialization, one `Value` at a time.
        report_json
            .as_object_mut()
            .context("AISimulate replay report did not serialize as an object")?
            .insert("per_request".to_string(), serde_json::Value::Null);
    }
    let report_json = ReportWithRecords {
        report: &report_json,
        records: &report.per_request,
    };
    if let Some(artifacts) = artifacts {
        let output = serde_json::json!({
            "report": report_json,
            "artifacts": artifacts,
        });
        return serde_json::to_string(&output).context("serializing AISimulate replay output");
    }
    serde_json::to_string(&report_json).context("serializing AISimulate replay output")
}

/// The report object with its `per_request` placeholder serialized as the
/// records, each converted to `Value` on its own. The output equals inserting
/// `serde_json::to_value(records)` and serializing the whole `Value`, without a
/// `Value` tree for every record at once.
struct ReportWithRecords<'a> {
    report: &'a serde_json::Value,
    records: &'a [crate::replay::PerRequestRecord],
}

impl serde::Serialize for ReportWithRecords<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let serde_json::Value::Object(object) = self.report else {
            return self.report.serialize(serializer);
        };
        let mut map = serializer.serialize_map(Some(object.len()))?;
        for (key, value) in object {
            if key == "per_request" {
                map.serialize_entry(key, &RecordValues(self.records))?;
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

struct RecordValues<'a>(&'a [crate::replay::PerRequestRecord]);

impl serde::Serialize for RecordValues<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeSeq};

        let mut records = serializer.serialize_seq(Some(self.0.len()))?;
        for record in self.0 {
            records.serialize_element(&serde_json::to_value(record).map_err(S::Error::custom)?)?;
        }
        records.end()
    }
}

fn replay_python_error(error: anyhow::Error) -> PyErr {
    let exception = if error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<crate::replay::ReplayError>(),
            Some(crate::replay::ReplayError::ResourceLimited(_))
        )
    }) {
        PyMemoryError::new_err(format!("{error:#}"))
    } else {
        PyRuntimeError::new_err(format!("{error:#}"))
    };
    if let Some(failure) = error.downcast_ref::<ReplayCoverageFailure>() {
        Python::with_gil(|py| {
            let _ = exception
                .value(py)
                .setattr("fpm_query_coverage", failure.coverage.to_string());
        });
    }
    exception
}

/// Execute one canonical serialized ReplaySpec and return serialized report JSON.
#[pyfunction]
fn run_replay_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, false))
        .map_err(replay_python_error)
}

/// Execute one fixed aggregated ReplaySpec and return report plus parity artifacts.
#[pyfunction]
fn run_replay_with_artifacts_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, true))
        .map_err(replay_python_error)
}

#[pyfunction]
fn canonical_timing_config_json(payload: &str, worker_type: &str) -> PyResult<String> {
    let convert = || -> anyhow::Result<String> {
        let timing: AicTimingConfig = serde_json::from_str(payload)?;
        let role = serde_json::from_value(serde_json::json!(worker_type))?;
        let config = timing.estimator_request(role)?;
        config.validate()?;
        Ok(serde_json::to_string(&config)?)
    };
    convert().map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:#}")))
}

/// Share the Runner's resolved role descriptor with downstream adapters.
#[pyfunction]
#[pyo3(signature = (payload, startup_time=None))]
fn engine_launch_from_replay_role_json(
    payload: &str,
    startup_time: Option<f64>,
) -> PyResult<String> {
    let convert = || -> anyhow::Result<String> {
        let role: crate::replay::ReplayRoleConfig = serde_json::from_str(payload)?;
        let mut launch = crate::engine::EngineLaunchConfig {
            engine: role.rank,
            dp_size: role.dp_size,
            tensor_parallel_size: role.tensor_parallel_size as usize,
            num_gpu_blocks_is_explicit: role.num_gpu_blocks_is_explicit.unwrap_or(false),
            startup_time,
            ..Default::default()
        };
        if let crate::engine::TimingModelConfig::External { provider, config } =
            &launch.engine.timing_model
            && matches!(provider.as_str(), "aic" | "ais")
        {
            let timing: AicTimingConfig = serde_json::from_value(config.clone())?;
            let worker_type = serde_json::from_value(serde_json::to_value(launch.worker_type)?)?;
            let canonical = timing.estimator_request(worker_type)?;
            launch.gpu_memory_utilization = timing.gpu_memory_utilization;
            launch.mem_fraction_static = timing.mem_fraction_static;
            launch.free_gpu_memory_fraction = timing.free_gpu_memory_fraction;
            launch.cuda_graph_reserved_bytes = Some(timing.cuda_graph_reserved_bytes);
            launch.engine.timing_model = crate::engine::TimingModelConfig::External {
                provider: "ais".into(),
                config: serde_json::to_value(canonical)?,
            };
        }
        Ok(serde_json::to_string(&launch.normalized()?)?)
    };
    convert().map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:#}")))
}

/// AISimulate native runtime module.
#[pymodule]
fn _runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        engine_launch_from_replay_role_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(canonical_timing_config_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_replay_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_replay_with_artifacts_json, module)?)?;
    crate::perfmodel::register_python(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::engine::{EngineConfig, TimingModelConfig};
    use crate::replay::{
        ProviderSpec, ReplayAdapters, ReplayEngineConfig, ReplayRequest, ReplaySpec,
        ReplayTopology, WorkerPoolSpec,
    };

    use super::*;

    #[test]
    fn replay_coverage_errors_preserve_the_original_chain() {
        let original = || anyhow::anyhow!("original failure").context("AISimulate replay failed");
        let wrapped = replay_failure(original(), Ok(Some(serde_json::json!({}))));
        assert_eq!(
            format!("{wrapped:#}"),
            "AISimulate replay failed: original failure"
        );
        let missing = replay_failure(original(), Err(anyhow::anyhow!("coverage unavailable")));
        assert_eq!(
            format!("{missing:#}"),
            "FPM coverage diagnostics failed: coverage unavailable: AISimulate replay failed: original failure"
        );
    }

    #[test]
    fn replay_python_error_preserves_contextual_resource_failure() {
        pyo3::prepare_freethreaded_python();
        let error = anyhow::Error::new(crate::replay::ReplayError::ResourceLimited(
            "create exact report sample file: No space left on device".to_string(),
        ))
        .context("collecting replay report")
        .context("AISimulate replay failed");
        let expected_message = format!("{error:#}");

        Python::with_gil(|py| {
            let mapped = replay_python_error(error);
            assert!(mapped.is_instance_of::<PyMemoryError>(py));
            assert_eq!(mapped.value(py).to_string(), expected_message);
        });
    }

    #[test]
    fn replay_python_error_keeps_other_failures_as_runtime_errors() {
        pyo3::prepare_freethreaded_python();
        let error = anyhow::anyhow!("invalid replay input").context("AISimulate replay failed");
        let expected_message = format!("{error:#}");

        Python::with_gil(|py| {
            let mapped = replay_python_error(error);
            assert!(mapped.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(mapped.value(py).to_string(), expected_message);
        });
    }

    struct PowerTiming(TimingEvidenceSummary);

    impl TimingModel for PowerTiming {
        fn predict_prefill_ms(
            &self,
            _batch_size: usize,
            _mean_isl: usize,
            _mean_prefix: usize,
        ) -> Result<f64> {
            Ok(0.0)
        }

        fn predict_decode_ms(
            &self,
            _batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> Result<f64> {
            Ok(0.0)
        }

        fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn cached_prefix_trace_blocks_share_the_same_engine_blocks_as_one_token_blocks() {
        let engine_block_size = 64;
        // Number each request's engine-block sequence hashes in first-seen
        // order, so traces with different token values compare by sharing.
        let shared_blocks = |cached_prefix_tokens, shared_prefix_ratio, block_size| {
            let trace = Trace::synthetic(SyntheticTraceSpec {
                block_size,
                num_sessions: 3,
                turns_per_session: 1,
                input_tokens: LengthSpec {
                    mean: 1000,
                    stddev: 0.0,
                },
                output_tokens: LengthSpec {
                    mean: 1,
                    stddev: 0.0,
                },
                cached_prefix_tokens,
                shared_prefix_ratio,
                // Three sessions in two groups put at least two in one group.
                num_prefix_groups: 2,
                first_turn_arrivals: ArrivalSpec::Burst,
                inter_turn_delays: DelaySpec::None,
                seed: 0,
                arrival_seed: 42,
                images: None,
            })
            .unwrap();
            let mut ids = std::collections::HashMap::new();
            trace
                .sessions
                .iter()
                .map(|session| {
                    let tokens = session.turns[0].synthesize_tokens(block_size).unwrap();
                    crate::replay::loadgen::ReplayRequestHashes::from_tokens(
                        &tokens,
                        engine_block_size as u32,
                    )
                    .sequence_hashes
                    .into_iter()
                    .map(|hash| {
                        let next = ids.len();
                        *ids.entry(hash).or_insert(next)
                    })
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        for (cached_prefix_tokens, shared_prefix_ratio, expected_block_size) in
            [(512, 0.0, 64), (500, 0.0, 4), (0, 0.0, 64), (512, 0.72, 1)]
        {
            let block_size = synthetic_trace_block_size(
                cached_prefix_tokens,
                shared_prefix_ratio,
                engine_block_size,
            );
            assert_eq!(block_size, expected_block_size);
            assert_eq!(
                shared_blocks(cached_prefix_tokens, shared_prefix_ratio, block_size),
                shared_blocks(cached_prefix_tokens, shared_prefix_ratio, 1),
                "cached_prefix_tokens={cached_prefix_tokens} \
                 shared_prefix_ratio={shared_prefix_ratio}"
            );
        }
    }

    #[test]
    fn warmup_requires_a_boolean_and_seeded_snapshot_at_the_native_boundary() {
        let base = serde_json::json!({
            "source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": "unused", "trace_format": "weka", "agentic_lanes": 1,
        });
        assert!(
            !serde_json::from_value::<RuntimeTraffic>(base.clone())
                .unwrap()
                .agentic_warmup
        );
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!("true"),
            serde_json::json!({}),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_warmup"] = invalid;
            assert!(serde_json::from_value::<RuntimeTraffic>(traffic).is_err());
        }
        let mut traffic = base;
        traffic["agentic_warmup"] = serde_json::json!(true);
        let traffic = serde_json::from_value::<RuntimeTraffic>(traffic).unwrap();
        assert!(
            build_runtime_input(traffic, 64)
                .err()
                .unwrap()
                .to_string()
                .contains("agentic_warmup requires agentic_snapshot")
        );
    }

    #[test]
    fn profile_options_are_opt_in_and_strict_at_the_native_boundary() {
        let base = serde_json::json!({
            "source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": "unused", "trace_format": "weka", "agentic_lanes": 1,
            "agentic_snapshot": {"seed": 42},
        });
        assert!(
            serde_json::from_value::<RuntimeTraffic>(base.clone())
                .unwrap()
                .agentic_profile
                .is_none()
        );
        let mut traffic = base.clone();
        traffic["agentic_profile"] = serde_json::json!({});
        let profile = serde_json::from_value::<RuntimeTraffic>(traffic.clone())
            .unwrap()
            .agentic_profile
            .unwrap();
        assert_eq!(profile.duration_seconds, 3600.0);
        assert_eq!(profile.response_grace_seconds, 30.0);
        assert_eq!(profile.cancel_drain_seconds, 10.0);
        assert_eq!(profile.tree_idle_cap_seconds, 300.0);
        assert_eq!(profile.global_idle_cap_seconds, 10.0);
        for invalid in [
            serde_json::json!(true),
            serde_json::json!({"duration_seconds": true}),
            serde_json::json!({"duration_seconds": "1"}),
            serde_json::json!({"unexpected": 1}),
        ] {
            traffic["agentic_profile"] = invalid;
            assert!(serde_json::from_value::<RuntimeTraffic>(traffic.clone()).is_err());
        }
        for invalid in [
            serde_json::json!({"duration_seconds": 0}),
            serde_json::json!({"response_grace_seconds": -1}),
            serde_json::json!({"cancel_drain_seconds": -1}),
            serde_json::json!({"tree_idle_cap_seconds": 0}),
            serde_json::json!({"global_idle_cap_seconds": 0}),
        ] {
            traffic["agentic_profile"] = invalid;
            let traffic = serde_json::from_value::<RuntimeTraffic>(traffic.clone()).unwrap();
            assert!(
                build_runtime_input(traffic, 64)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("agentic_profile")
            );
        }
        for (field, value, message) in [
            (
                "agentic_snapshot",
                serde_json::Value::Null,
                "requires agentic_snapshot",
            ),
            (
                "max_sim_time_ms",
                serde_json::json!(1.0),
                "cannot be combined",
            ),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_profile"] = serde_json::json!({});
            traffic[field] = value;
            let traffic = serde_json::from_value::<RuntimeTraffic>(traffic).unwrap();
            assert!(
                build_runtime_input(traffic, 64)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains(message)
            );
        }
        let payload = serde_json::json!({
            "spec": {
                "version": 1,
                "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
                "max_sim_time_ms": 1.0,
                "requests": []
            },
            "traffic": {"agentic_profile": {}, "source_type": "trace"}
        });
        assert!(
            execute_json(&payload.to_string(), false)
                .unwrap_err()
                .to_string()
                .contains("agentic_profile cannot be combined with ReplaySpec.max_sim_time_ms")
        );
    }

    #[test]
    fn image_traffic_is_rejected_with_an_encoder_pool() {
        let payload = serde_json::json!({
            "spec": {
                "version": 1,
                "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
                "encoder": {
                    "instances": 1, "max_batch": 1, "gpus_per_instance": 1, "images_per_request": 1,
                    "shape": {"sequences": 1, "patch_tokens": 1, "transformer_tokens": 1, "output_tokens": 1},
                    "preprocess_ms_per_image": 0.0, "transfer_bytes_per_image": 0,
                    "transfer_bandwidth_gb_s": 1.0
                },
                "requests": []
            },
            "traffic": {"source_type": "synthetic", "image_count": 1}
        });
        assert!(
            execute_json(&payload.to_string(), false)
                .unwrap_err()
                .to_string()
                .contains("image traffic cannot be combined with an encoder pool")
        );
    }

    #[test]
    fn snapshot_options_are_strict_at_the_native_json_boundary() {
        let base = serde_json::json!({
            "source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": "unused", "trace_format": "weka", "agentic_lanes": 1,
        });
        for invalid in [
            serde_json::json!({}),
            serde_json::json!({"seed": true}),
            serde_json::json!({"seed": -1}),
            serde_json::json!({"seed": 1.0}),
            serde_json::json!({"seed": "42"}),
            serde_json::json!({"seed": 42, "extra": 0}),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_snapshot"] = invalid;
            assert!(serde_json::from_value::<RuntimeTraffic>(traffic).is_err());
        }
        for (field, invalid) in [
            ("source_type", serde_json::json!("synthetic")),
            ("load_type", serde_json::json!("concurrency")),
            ("agentic_lanes", serde_json::json!(0)),
            ("agentic_lanes", serde_json::Value::Null),
            ("replay_concurrency", serde_json::json!(1)),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_snapshot"] = serde_json::json!({"seed": u64::MAX});
            traffic[field] = invalid;
            let traffic = serde_json::from_value::<RuntimeTraffic>(traffic).unwrap();
            assert!(
                build_runtime_input(traffic, 64)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("agentic_snapshot requires")
            );
        }
    }

    #[test]
    fn public_agentic_json_host_offload_validates_the_whole_deployment() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("original-weka.jsonl");
        std::fs::write(&path, serde_json::json!({
            "id": "play", "models": ["model"], "block_size": 4, "hash_id_scope": "local",
            "requests": [{"t": 0.0, "type": "s", "model": "model", "in": 4, "out": 1, "hash_ids": [1]}]
        }).to_string()).unwrap();
        let rank = serde_json::json!({
            "backend": "vllm", "block_size": 4, "num_gpu_blocks": 16,
            "kv_cache_bytes_per_token": 16,
            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
        });
        let with_host = |scope: Option<&str>| {
            let mut rank = rank.clone();
            if let Some(scope) = scope {
                rank["native_host_offload"] = serde_json::json!({
                    "scope": scope, "num_host_blocks": 8, "kv_layout_id": "original-test-layout"
                });
            }
            serde_json::json!({"tensor_parallel_size": 2, "rank": rank})
        };
        let payload = |topology, engine| {
            serde_json::json!({
                "spec": {"version": 1, "topology": topology, "engine": engine, "requests": []},
                "traffic": {
                    "source_type": "trace", "load_type": "trace_timestamps", "trace_format": "weka",
                    "trace_path": path, "trace_block_size": 4, "execution_model": "model"
                }
            })
        };
        let run = |payload: &serde_json::Value| -> serde_json::Value {
            serde_json::from_str(&execute_json(&payload.to_string(), false).unwrap()).unwrap()
        };
        let scopes = [None, Some("dp_rank_local"), Some("cluster_shared")];
        for scope in scopes {
            let agg = payload(
                serde_json::json!({"kind": "aggregated", "workers": {"initial_workers": 1}}),
                with_host(scope),
            );
            let report = run(&agg);
            assert_eq!(report["completed_requests"], 1);
            for other in scopes {
                let pd = payload(
                    serde_json::json!({"kind": "disaggregated", "prefill": {"initial_workers": 1}, "decode": {"initial_workers": 1}}),
                    serde_json::json!({"prefill": with_host(scope), "decode": with_host(other)}),
                );
                let report = run(&pd);
                assert_eq!(report["completed_requests"], 1, "{scope:?}/{other:?}");
                if scope.is_none() && other.is_none() {
                    continue;
                }
                for (pointer, value, message) in [
                    (
                        "/spec/topology/prefill/initial_workers",
                        serde_json::json!(2),
                        "one aggregated worker or one prefill and one decode worker",
                    ),
                    (
                        "/spec/topology/decode/initial_workers",
                        serde_json::json!(2),
                        "one aggregated worker or one prefill and one decode worker",
                    ),
                    ("/spec/engine/prefill/dp_size", serde_json::json!(2), "DP=1"),
                    ("/spec/engine/decode/dp_size", serde_json::json!(2), "DP=1"),
                    (
                        "/spec/engine/prefill/rank/backend",
                        serde_json::json!("sglang"),
                        "backend=vllm",
                    ),
                    (
                        "/spec/engine/decode/rank/backend",
                        serde_json::json!("sglang"),
                        "backend=vllm",
                    ),
                    (
                        "/spec/engine/decode/rank/aic_nextn",
                        serde_json::json!(1),
                        "speculative decoding disabled",
                    ),
                    (
                        "/spec/engine/decode/rank/g3_offload",
                        serde_json::json!({"scope": "worker_local", "num_g3_blocks": 8}),
                        "G3",
                    ),
                ] {
                    let mut invalid = pd.clone();
                    let (parent, key) = pointer.rsplit_once('/').unwrap();
                    invalid.pointer_mut(parent).unwrap()[key] = value;
                    let error = format!(
                        "{:#}",
                        execute_json(&invalid.to_string(), false).unwrap_err()
                    );
                    assert!(
                        error.contains(message),
                        "{scope:?}/{other:?} {pointer}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn public_live_agentic_warmup_keeps_only_formal_requests_in_the_report() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("live-weka.jsonl");
        let rows = (0..3)
            .map(|index| {
                serde_json::json!({
                    "t": index as f64 * 0.1, "type": "s", "model": "model",
                    "in": 4, "out": 3, "hash_ids": [index + 1], "api_time": 0.001
                })
            })
            .collect::<Vec<_>>();
        std::fs::write(
            &path,
            serde_json::json!({
                "id": "play", "models": ["model"], "block_size": 4,
                "hash_id_scope": "local", "requests": rows
            })
            .to_string(),
        )
        .unwrap();
        for backend in ["vllm", "sglang"] {
            let payload = serde_json::json!({
                "spec": {
                    "version": 1, "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
                    "engine": {"rank": {"backend": backend, "block_size": 4, "num_gpu_blocks": 64,
                        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
                    "requests": []
                },
                "traffic": {"source_type": "trace", "load_type": "trace_timestamps", "trace_format": "weka",
                    "trace_path": path, "trace_block_size": 4, "execution_model": "model",
                    "agentic_lanes": 1, "agentic_snapshot": {"seed": 42}, "agentic_warmup": true,
                    "agentic_warmup_policy": "live_trajectory_v1", "agentic_profile": {"duration_seconds": 0.01}}
            });
            let report: serde_json::Value =
                serde_json::from_str(&execute_json(&payload.to_string(), false).unwrap()).unwrap();
            let phases = &report["agentic_phases"];
            assert_eq!(phases["phase"], "profile", "{backend}: {report}");
            assert_eq!(phases["warmup_policy"], "live_trajectory_v1");
            assert_eq!(phases["lanes"][0]["warmup_completed"], 10);
            assert_eq!(phases["lanes"][0]["primers_completed"], 1);
            let formal = report["per_request"].as_array().unwrap();
            assert!(!formal.is_empty());
            assert_eq!(
                formal.len() as u64,
                report["agentic_profile"]["issued_requests"]
                    .as_u64()
                    .unwrap()
            );
            assert!(report["completed_requests"].as_u64().unwrap() > 0);
            let outcomes = report["agentic_play_outcomes"].as_array().unwrap();
            assert_eq!(
                outcomes.len() as u64,
                report["total_trajectories"].as_u64().unwrap()
            );
            assert_eq!(
                outcomes.len() as u64,
                report["agentic_profile"]["plays_started"].as_u64().unwrap()
            );
            for outcome in outcomes {
                for field in ["causal_terminal_ms", "settled_at_ms"] {
                    assert!(
                        outcome[field].as_f64().is_none_or(|value| value >= 0.0),
                        "{backend}: {outcome}"
                    );
                }
            }
            let mut invalid = payload.clone();
            invalid["traffic"]
                .as_object_mut()
                .unwrap()
                .remove("agentic_profile");
            assert!(
                format!(
                    "{:#}",
                    execute_json(&invalid.to_string(), false).unwrap_err()
                )
                .contains("live_trajectory_v1 requires agentic_warmup and agentic_profile")
            );
        }
    }

    #[test]
    fn live_zero_time_warmup_does_not_poison_the_formal_progress_guard() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mixed-duration.jsonl");
        let rows = [("zero", 1), ("positive", 3)].into_iter().map(|(id, output)| serde_json::json!({
            "id": id, "models": ["model"], "block_size": 4, "hash_id_scope": "local",
            "requests": [{"t": 0.0, "type": "s", "model": "model", "in": 4, "out": output, "hash_ids": [1]}]
        }).to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&path, rows).unwrap();
        let payload = serde_json::json!({
            "spec": {
                "version": 1, "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
                "engine": {"rank": {"backend": "vllm", "block_size": 4, "num_gpu_blocks": 64,
                    "timing_model": {"type": "fixed", "prefill_ms": 0.0, "decode_ms": 1.0}}},
                "requests": []
            },
            "traffic": {"source_type": "trace", "load_type": "trace_timestamps", "trace_format": "weka",
                "trace_path": path, "trace_block_size": 4, "execution_model": "model",
                "agentic_lanes": 1, "agentic_snapshot": {"seed": 42}, "agentic_warmup": true,
                "agentic_warmup_policy": "live_trajectory_v1", "agentic_profile": {"duration_seconds": 0.01}}
        });
        let report: serde_json::Value =
            serde_json::from_str(&execute_json(&payload.to_string(), false).unwrap()).unwrap();
        assert_eq!(report["agentic_phases"]["profile_start_ms"], 0.0);
        assert_eq!(report["agentic_phases"]["lanes"][0]["warmup_completed"], 10);
        assert!(report["duration_ms"].as_f64().unwrap() >= 10.0);
        assert!(report["completed_requests"].as_u64().unwrap() >= 2);
    }

    #[test]
    fn public_agentic_hbm_speculation_completes_dependency_workloads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sd-weka.jsonl");
        std::fs::write(
            &path,
            serde_json::json!({
                "id": "play", "models": ["model"], "block_size": 4, "hash_id_scope": "local",
                "requests": [
                    {"t": 0.0, "type": "s", "model": "model", "in": 4, "out": 8, "hash_ids": [1]},
                    {"t": 1.0, "type": "s", "model": "model", "in": 8, "out": 8, "hash_ids": [1, 2]}
                ]
            })
            .to_string(),
        )
        .unwrap();
        for backend in ["vllm", "sglang"] {
            for disagg in [false, true] {
                let rank = serde_json::json!({
                    "backend": backend, "block_size": 4, "num_gpu_blocks": 64,
                    "aic_nextn": 3, "aic_nextn_accept_rates": "1,0.78,0",
                    "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                });
                let (topology, engine) = if disagg {
                    (
                        serde_json::json!({"kind": "disaggregated", "prefill": {"initial_workers": 1}, "decode": {"initial_workers": 1}}),
                        serde_json::json!({"prefill": {"rank": rank}, "decode": {"rank": rank}}),
                    )
                } else {
                    (
                        serde_json::json!({"kind": "aggregated", "workers": {"initial_workers": 1}}),
                        serde_json::json!({"rank": rank}),
                    )
                };
                let payload = serde_json::json!({
                    "spec": {"version": 1, "topology": topology, "engine": engine, "requests": []},
                    "traffic": {
                        "source_type": "trace", "load_type": "trace_timestamps", "trace_format": "weka",
                        "trace_path": path, "trace_block_size": 4, "execution_model": "model", "agentic_lanes": 1
                    }
                });
                let run = || execute_json(&payload.to_string(), false).unwrap();
                let report: serde_json::Value = serde_json::from_str(&run()).unwrap();
                assert_eq!(report["completed_requests"], 2, "{backend}/{disagg}");
                assert_eq!(report["total_output_tokens"], 16, "{backend}/{disagg}");
                assert_eq!(report["agentic_qualification"], "functional_only");
                assert!(report["agentic_lifecycle_digest"].is_string());
                // The public path must actually schedule sampled SD progress,
                // rather than merely accept and silently discard the settings.
                let mut ordinary_payload = payload.clone();
                let paths = if disagg {
                    vec!["/spec/engine/prefill/rank", "/spec/engine/decode/rank"]
                } else {
                    vec!["/spec/engine/rank"]
                };
                for path in paths {
                    let rank = ordinary_payload
                        .pointer_mut(path)
                        .unwrap()
                        .as_object_mut()
                        .unwrap();
                    rank.remove("aic_nextn");
                    rank.remove("aic_nextn_accept_rates");
                }
                let ordinary: serde_json::Value = serde_json::from_str(
                    &execute_json(&ordinary_payload.to_string(), false).unwrap(),
                )
                .unwrap();
                assert!(
                    report["duration_ms"].as_f64().unwrap()
                        < ordinary["duration_ms"].as_f64().unwrap()
                );
                let repeated: serde_json::Value = serde_json::from_str(&run()).unwrap();
                for field in [
                    "duration_ms",
                    "agentic_lifecycle_digest",
                    "total_output_tokens",
                ] {
                    assert_eq!(report[field], repeated[field], "{backend}/{disagg}/{field}");
                }
            }
        }
    }

    #[test]
    fn public_agentic_json_rejects_unqualified_modes_without_restricting_standard_dynamo() {
        let directory = tempfile::tempdir().unwrap();
        let dynamo = serde_json::json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": 10,
            "agent_context": {"session_id": "session"},
            "request": {
                "request_id": "root", "model": "model", "output_tokens": 1,
                "request_received_ms": 0, "total_time_ms": 10,
                "replay": {"trace_block_size": 4, "input_length": 4, "input_sequence_hashes": [1]}
            }
        });
        let mut standard_dynamo = dynamo.clone();
        standard_dynamo
            .as_object_mut()
            .unwrap()
            .remove("agent_context");
        let fixtures = [
            (
                "weka",
                vec![serde_json::json!({
                    "id": "play", "models": ["model"], "block_size": 4, "hash_id_scope": "local",
                    "requests": [{"t": 0.0, "type": "s", "model": "model", "in": 4, "out": 1, "hash_ids": [1]}]
                })],
                true,
            ),
            (
                "agentic_mooncake",
                vec![
                    serde_json::json!({
                        "schema": "dynamo.agentic_mooncake", "version": 2,
                        "block_size": 4, "hash_id_scope": "local",
                        "source": {"format": "test", "digest": "qualification"}
                    }),
                    serde_json::json!({
                        "request_id": "root", "play_id": "play", "session_id": "session", "model": "model",
                        "input_length": 4, "output_length": 1, "hash_ids": [1], "not_before_ms": 0.0
                    }),
                ],
                true,
            ),
            ("dynamo", vec![dynamo], true),
            ("dynamo", vec![standard_dynamo], false),
        ];
        for (format, rows, agentic) in fixtures {
            let path = directory.path().join(format!("{format}-{agentic}.jsonl"));
            std::fs::write(
                &path,
                rows.iter()
                    .map(|row| format!("{row}\n"))
                    .collect::<String>(),
            )
            .unwrap();
            for (options, message) in
                [(serde_json::json!({"backend": "trtllm"}), "vLLM and SGLang")]
            {
                let mut rank = serde_json::json!({
                    "backend": "vllm", "block_size": 4, "num_gpu_blocks": 16,
                    "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                });
                rank.as_object_mut()
                    .unwrap()
                    .extend(options.as_object().unwrap().clone());
                let payload = serde_json::json!({
                    "spec": {
                        "version": 1,
                        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1, "startup_delay_ms": 0.0}},
                        "engine": {"rank": rank},
                        "requests": []
                    },
                    "traffic": {
                        "source_type": "trace", "load_type": "trace_timestamps",
                        "trace_format": format, "trace_path": path,
                        "trace_block_size": 4, "execution_model": "model"
                    }
                });
                // No lane flag: Dynamo agentic detection must follow loaded content.
                let result = execute_json(&payload.to_string(), false);
                if agentic {
                    let error = format!("{:#}", result.unwrap_err());
                    assert!(error.contains(message), "{format}: {error}");
                    // Public P/D must validate both roles, including a decode
                    // role whose invalid configuration is absent from prefill.
                    for invalid_role in ["prefill", "decode"] {
                        let mut disagg = payload.clone();
                        disagg["spec"]["topology"] = serde_json::json!({
                            "kind": "disaggregated",
                            "prefill": {"initial_workers": 1},
                            "decode": {"initial_workers": 1}
                        });
                        let valid = serde_json::json!({
                            "backend": "vllm", "block_size": 4, "num_gpu_blocks": 16,
                            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                        });
                        disagg["spec"]["engine"] = serde_json::json!({
                            "prefill": {"rank": valid}, "decode": {"rank": valid}
                        });
                        disagg["spec"]["engine"][invalid_role]["rank"] = rank.clone();
                        let error = format!(
                            "{:#}",
                            execute_json(&disagg.to_string(), false).unwrap_err()
                        );
                        assert!(error.contains(message), "{format}/{invalid_role}: {error}");
                    }
                } else {
                    let report: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
                    assert_eq!(report["completed_requests"], 1);
                    assert!(report.get("agentic_qualification").is_none());
                }
            }
            // Exercise all three importers through the actual native P/D
            // entrypoint. Standard Dynamo remains a non-agentic workload.
            for backend in ["vllm", "sglang"] {
                let rank = serde_json::json!({
                    "backend": backend, "block_size": 4, "num_gpu_blocks": 16,
                    "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                });
                let mut payload = serde_json::json!({
                    "spec": {
                        "version": 1,
                        "topology": {
                            "kind": "disaggregated",
                            "prefill": {"initial_workers": 1},
                            "decode": {"initial_workers": 1}
                        },
                        "engine": {"prefill": {"rank": rank}, "decode": {"rank": rank}},
                        "requests": []
                    },
                    "traffic": {
                        "source_type": "trace", "load_type": "trace_timestamps",
                        "trace_format": format, "trace_path": path,
                        "trace_block_size": 4, "execution_model": "model"
                    }
                });
                if agentic {
                    for invalid_roles in
                        [vec!["prefill"], vec!["decode"], vec!["prefill", "decode"]]
                    {
                        let mut invalid = payload.clone();
                        for role in ["prefill", "decode"] {
                            invalid["spec"]["engine"][role]["rank"]["timing_model"] = serde_json::json!({
                                "type": "external", "provider": "aic",
                                "config": {"model": if invalid_roles.contains(&role) { "different-model" } else { "model" }, "backend": backend, "system": "test-system", "tp": 1}
                            });
                        }
                        let error = format!(
                            "{:#}",
                            execute_json(&invalid.to_string(), false).unwrap_err()
                        );
                        assert!(
                            error.contains("must match agentic execution model"),
                            "{format}/{backend}/{invalid_roles:?}: {error}"
                        );
                    }
                } else {
                    // Standard Dynamo still reaches ordinary capacity/provider
                    // validation instead of acquiring an agentic model gate.
                    let mut ordinary = payload.clone();
                    ordinary["traffic"]["load_type"] = serde_json::json!("kv_capacity_fraction");
                    ordinary["traffic"]["kv_load_ratio"] = serde_json::json!(0);
                    let error = format!(
                        "{:#}",
                        execute_json(&ordinary.to_string(), false).unwrap_err()
                    );
                    assert!(error.contains("KV load ratio must be positive"), "{error}");
                    ordinary = payload.clone();
                    ordinary["spec"]["engine"]["decode"]["rank"]["timing_model"] = serde_json::json!({
                        "type": "external", "provider": "aic", "config": {"model": "different-model"}
                    });
                    let error = format!(
                        "{:#}",
                        execute_json(&ordinary.to_string(), false).unwrap_err()
                    );
                    assert!(
                        error.contains("invalid AIC timing provider configuration"),
                        "{error}"
                    );
                }
                for warmup in [false, true] {
                    if warmup {
                        if !agentic {
                            continue;
                        }
                        payload["traffic"]["agentic_lanes"] = serde_json::json!(1);
                        payload["traffic"]["agentic_snapshot"] = serde_json::json!({"seed": 0});
                        payload["traffic"]["agentic_warmup"] = serde_json::json!(true);
                    }
                    let report: serde_json::Value = serde_json::from_str(
                        &execute_json(&payload.to_string(), false).unwrap_or_else(|error| {
                            panic!("{format}/{backend}/{warmup}: {error:#}")
                        }),
                    )
                    .unwrap();
                    assert_eq!(report["completed_requests"], 1);
                    assert_eq!(report.get("agentic_qualification").is_some(), agentic);
                    if warmup {
                        assert_eq!(report["agentic_phases"]["lanes"][0]["warmup_completed"], 10);
                        assert!(
                            report["agentic_phases"]["profile_start_ms"]
                                .as_f64()
                                .is_some()
                        );
                    }
                }
            }
        }
    }

    #[pyclass]
    struct DecodeCoordinateProbe;

    #[pymethods]
    impl DecodeCoordinateProbe {
        fn predict_decode_latency(
            &self,
            batch_size: u32,
            mean_context_length: u32,
            _osl: u32,
        ) -> u64 {
            u64::from(batch_size) * u64::from(mean_context_length + 1)
        }

        fn predict_decode_latency_total(&self, _batch_size: u32, total_past_kv_tokens: u32) -> u64 {
            u64::from(total_past_kv_tokens)
        }

        #[allow(clippy::too_many_arguments)]
        fn run_static_per_op(
            &self,
            batch_size: u32,
            beam_width: u32,
            isl: u32,
            osl: u32,
            prefix: u32,
            seq_imbalance_correction_scale: f64,
            gen_seq_imbalance_correction_scale: f64,
            mode: &str,
            stride: u32,
        ) -> (
            Vec<(String, f64, f64, String)>,
            Vec<(String, f64, f64, String)>,
        ) {
            let _ = (
                beam_width,
                osl,
                prefix,
                seq_imbalance_correction_scale,
                gen_seq_imbalance_correction_scale,
                mode,
                stride,
            );
            let latency_ms = f64::from(batch_size) * f64::from(isl + 1);
            (
                Vec::new(),
                vec![("decode".into(), latency_ms, 0.0, "test".into())],
            )
        }
    }

    /// Encodes each prefill query's coordinates as `a * 10^6 + b * 10^3 + c`.
    #[pyclass]
    struct PrefillCoordinateProbe;

    #[pymethods]
    impl PrefillCoordinateProbe {
        /// The FPM engine's mean API queries `(bs, bs * (isl - prefix), bs * prefix)`.
        fn predict_prefill_latency(&self, bs: u64, isl: u64, prefix: u64) -> u64 {
            bs * 1_000_000 + bs * (isl - prefix) * 1_000 + bs * prefix
        }

        fn predict_prefill_latency_total(
            &self,
            batch_size: u64,
            total_prefill_tokens: u64,
            total_kv_read_tokens: u64,
        ) -> u64 {
            batch_size * 1_000_000 + total_prefill_tokens * 1_000 + total_kv_read_tokens
        }

        /// Op-level context evidence at the requested per-request shape.
        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        fn run_static_per_op(
            &self,
            batch_size: u32,
            beam_width: u32,
            isl: u32,
            osl: u32,
            prefix: u32,
            seq_imbalance_correction_scale: f64,
            gen_seq_imbalance_correction_scale: f64,
            mode: &str,
            stride: u32,
        ) -> (Vec<TestPerOpEvidence>, Vec<TestPerOpEvidence>) {
            let latency_ms = f64::from(batch_size) * 1e6 + f64::from(isl) * 1e3 + f64::from(prefix);
            (
                vec![("gemm".into(), latency_ms, 0.0, "silicon".into())],
                Vec::new(),
            )
        }
    }

    #[pyclass]
    #[derive(Default)]
    struct PerOpEvidenceProbe {
        calls: std::sync::atomic::AtomicUsize,
    }

    type TestPerOpEvidence = (String, f64, f64, String);

    #[pymethods]
    impl PerOpEvidenceProbe {
        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        fn run_static_per_op(
            &self,
            batch_size: u32,
            beam_width: u32,
            isl: u32,
            osl: u32,
            prefix: u32,
            seq_imbalance_correction_scale: f64,
            gen_seq_imbalance_correction_scale: f64,
            mode: &str,
            stride: u32,
        ) -> (Vec<TestPerOpEvidence>, Vec<TestPerOpEvidence>) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if batch_size == 99 {
                return (Vec::new(), Vec::new());
            }
            match mode {
                "static_ctx" => (
                    vec![
                        ("gemm".into(), 8.0, 3_200.0, "silicon".into()),
                        ("attention".into(), 2.0, 0.0, "empirical".into()),
                    ],
                    Vec::new(),
                ),
                "static_gen" => (
                    Vec::new(),
                    vec![("gemm".into(), 4.0, 1_600.0, "silicon".into())],
                ),
                unexpected => panic!("unexpected mode {unexpected}"),
            }
        }
    }

    fn python_timing_model(engine: Py<PyAny>, use_fpm_decode_totals: bool) -> AicTimingModel {
        AicTimingModel {
            engine,
            diagnostic_model: None,
            fpm_model: None,
            corrections: None,
            decoder_replay: false,
            phase_provider: AicPhaseProvider::Python,
            use_fpm_decode_totals,
            fpm_decode_kv_ceiling: None,
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            fpm_evidence: Mutex::default(),
            fpm_iteration_evidence: Mutex::default(),
            phase_cache: quick_cache::sync::Cache::new(128),
            vision: None,
            iteration: None,
        }
    }

    #[test]
    fn native_phase_evidence_matches_python_bridge() {
        use crate::operators::{FpmForwardOp, FpmPhase, op::Op, util_empirical::ProvenanceTier};
        use crate::perfmodel::engine::spec::EngineSpec;

        let required = std::env::var_os("AIC_REQUIRE_EMBEDDED_ROUND_TRIP").is_some();
        if let Err(error) =
            Python::with_gil(|py| py.import("aisimulate_core.sdk.engine").map(|_| ()))
        {
            assert!(
                !required,
                "native_phase_evidence_matches_python_bridge: \
                 AIC_REQUIRE_EMBEDDED_ROUND_TRIP is set but \
                 `aisimulate_core.sdk.engine` is not importable: {error}. \
                 Install the Python SDK and set PYTHONPATH for the embedded interpreter."
            );
            eprintln!(
                "native_phase_evidence_matches_python_bridge: SKIP — \
                 `aisimulate_core.sdk.engine` is not importable: {error}. \
                 Set AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1 + PYTHONPATH to enforce."
            );
            return;
        }

        let mut config = aic_config();
        config.model = "Qwen/Qwen3.8-2.4T-A95B-FP8".into();
        config.system = "gb300".into();
        config.backend = "sglang".into();
        config.backend_version = Some("next".into());
        config.tp = 16;
        config.moe_tp_size = Some(4);
        config.moe_ep_size = Some(4);
        config.systems_paths = vec![
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems"),
        ];
        let mut native =
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false).unwrap();
        native.diagnostic_model = None;
        let raw = match &native.phase_provider {
            AicPhaseProvider::Native(engine) => Arc::clone(engine),
            AicPhaseProvider::Fitted
            | AicPhaseProvider::Python
            | AicPhaseProvider::NativeEntries(_) => unreachable!(),
        };
        let bridge = Python::with_gil(|py| python_timing_model(native.engine.clone_ref(py), false));
        let check =
            |native: &AicTimingModel, bridge: &AicTimingModel, args| {
                let (batch, isl, osl, prefix, mode) = args;
                raw.database().note_provenance(ProvenanceTier::Empirical);
                let actual = native.predict_phase_evidence(batch, isl, osl, prefix, mode);
                let provenance = raw.last_provenance();
                raw.database().note_provenance(ProvenanceTier::Empirical);
                let expected = bridge.predict_phase_evidence(batch, isl, osl, prefix, mode);
                assert_eq!(provenance, raw.last_provenance());
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        let (a, b) = (actual.to_phase(), expected.to_phase());
                        let (a, b) = (a.as_ref(), b.as_ref());
                        assert_eq!(a, b);
                        let bits =
                            |phase: &TimingPhaseEvidence| {
                                std::iter::once((
                                    phase.latency_ms,
                                    phase.energy_wms,
                                    phase.covered_latency_ms,
                                ))
                                .chain(phase.operations.iter().map(|op| {
                                    (op.latency_ms, op.energy_wms, op.covered_latency_ms)
                                }))
                                .map(|(latency, energy, covered)| {
                                    (
                                        latency.to_bits(),
                                        energy.map(f64::to_bits),
                                        covered.to_bits(),
                                    )
                                })
                                .collect::<Vec<_>>()
                            };
                        assert_eq!(bits(a), bits(b));
                    }
                    (Err(actual), Err(expected)) => {
                        assert_eq!(actual.to_string(), expected.to_string())
                    }
                    mismatch => panic!("native/bridge mismatch: {mismatch:?}"),
                }
            };
        for args in [
            (0, 128, 1, 0, "static_ctx"),
            (4, 1024, 1, 128, "static_ctx"),
            (4, 128, 1, 128, "static_ctx"),
            (1, 128, 2, 0, "static_gen"),
            (32, 4096, 2, 0, "static_gen"),
            (7, 1777, 3, 0, "invalid-mode"),
        ] {
            check(&native, &bridge, args);
            check(&native, &bridge, args); // Cache hits must retain the same behavior.
        }

        // Exercise the shipping adapter through its public timing entrypoints,
        // including accumulation on cache hits and changes in query geometry.
        for (prefill, batch, length, prefix) in [
            (true, 4, 1024, 128),
            (true, 4, 1024, 128),
            (true, 4, 1536, 256),
            (false, 1, 129, 0),
            (false, 1, 129, 0),
            (false, 1, 257, 0),
        ] {
            let key = if prefill {
                (batch as u32, length as u32, 1, prefix as u32, true)
            } else {
                (batch as u32, length as u32 - 1, 2, 0, false)
            };
            let cached = native.phase_cache.get(&key);
            let predict = |timing: &AicTimingModel| {
                if prefill {
                    timing.predict_prefill_ms(batch, length, prefix)
                } else {
                    timing.predict_decode_ms(batch, batch * length, length, 16384)
                }
                .unwrap()
            };
            assert_eq!(predict(&native).to_bits(), predict(&bridge).to_bits());
            if let Some(cached) = cached {
                assert!(Arc::ptr_eq(&cached, &native.phase_cache.get(&key).unwrap()));
            }
            let actual = native.evidence_summary().unwrap();
            let expected = bridge.evidence_summary().unwrap();
            assert_eq!(actual, expected);
            for (a, b) in [
                (&actual.prefill, &expected.prefill),
                (&actual.decode, &expected.decode),
            ] {
                let bits = |phase: &TimingPhaseEvidence| {
                    std::iter::once((phase.latency_ms, phase.energy_wms, phase.covered_latency_ms))
                        .chain(
                            phase
                                .operations
                                .iter()
                                .map(|op| (op.latency_ms, op.energy_wms, op.covered_latency_ms)),
                        )
                        .map(|(latency, energy, covered)| {
                            (
                                latency.to_bits(),
                                energy.map(f64::to_bits),
                                covered.to_bits(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(a), bits(b));
            }
        }

        // Exercise a real missing-data error through both callers. Only the
        // test engine is synthetic; the production constructor is unchanged.
        let missing_op = |phase| {
            Op::FpmForward(FpmForwardOp {
                dcp_size: None,
                interpolation: Default::default(),
                name: "missing".into(),
                phase,
                model_path: "missing-test-model".into(),
                match_identity: vec!["missing".into(); 11],
                weight_bytes: 0.0,
                verify_width: 1,
                original_fmha_quant_mode: None,
                sol_ops: Vec::new(),
            })
        };
        let identity = serde_json::from_value(serde_json::json!({
            "schema_version": crate::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "missing-test-model", "system_name": "gb300", "backend": "sglang",
            "tp_size": 16, "pp_size": 1
        }))
        .unwrap();
        let missing = Arc::new(
            PerfEngine::build(
                EngineSpec::new(
                    identity,
                    vec![missing_op(FpmPhase::Prefill)],
                    vec![missing_op(FpmPhase::Decode)],
                ),
                Arc::clone(raw.database()),
            )
            .unwrap(),
        );
        let mut native = Python::with_gil(|py| {
            python_timing_model(
                Py::new(
                    py,
                    crate::AicEngine::from_shared_engine(Arc::clone(&missing)),
                )
                .unwrap()
                .into_any(),
                false,
            )
        });
        let bridge = Python::with_gil(|py| python_timing_model(native.engine.clone_ref(py), false));
        native.phase_provider = AicPhaseProvider::Native(missing);
        for mode in ["static_ctx", "static_gen"] {
            let error = native
                .predict_phase_evidence(1, 128, 2, 0, mode)
                .unwrap_err();
            assert!(
                error.to_string().contains("PerfDataNotAvailableError"),
                "{error}"
            );
            check(&native, &bridge, (1, 128, 2, 0, mode));
        }
    }

    #[test]
    fn canonical_timing_infers_topology_and_rejects_explicit_conflicts() {
        let mut engine = serde_json::json!({"rank": {"timing_model": {"type":"external", "provider":"aic", "config":{"tp":2, "attention_dp":4, "cp_size":8, "dcp":2}}}});
        infer_aic_timing_topology(&mut engine).unwrap();
        assert_eq!(engine["tensor_parallel_size"], 2);
        assert_eq!(engine["dp_size"], 4);
        assert_eq!(engine["prefill_context_parallel_size"], 8);
        let mut conflicting_cp = engine.clone();
        conflicting_cp["prefill_context_parallel_size"] = serde_json::json!(1);
        assert!(infer_aic_timing_topology(&mut conflicting_cp).is_err());
        engine["tensor_parallel_size"] = serde_json::json!(1);
        assert!(infer_aic_timing_topology(&mut engine).is_err());
    }

    #[test]
    fn estimator_request_carries_both_context_parallel_knobs() {
        // The native estimator path builds the engine from this request, so a
        // dropped knob would silently price a dcp=1 engine for a dcp=8 worker.
        let mut config = aic_config();
        config.cp_size = Some(2);
        config.dcp = Some(4);
        let request = config
            .estimator_request(ForwardPassWorkerType::Aggregated)
            .unwrap();
        assert_eq!(request.cp_size, Some(2));
        assert_eq!(request.dcp, Some(4));
        let unset = aic_config()
            .estimator_request(ForwardPassWorkerType::Aggregated)
            .unwrap();
        assert_eq!((unset.cp_size, unset.dcp), (None, None));
    }

    #[test]
    fn canonical_and_legacy_default_selection_are_distinct() {
        let mut config = aic_config();
        assert_eq!(
            config
                .estimator_request(ForwardPassWorkerType::Aggregated)
                .unwrap()
                .estimation_mode,
            EstimationMode::OpLevel
        );
        config.worker_type = Some(ForwardPassWorkerType::Aggregated);
        assert_eq!(
            config
                .estimator_request(ForwardPassWorkerType::Aggregated)
                .unwrap()
                .estimation_mode,
            EstimationMode::Auto
        );
        config.forward_model = Some("typo".into());
        assert!(
            config
                .estimator_request(ForwardPassWorkerType::Aggregated)
                .is_err()
        );
    }

    #[test]
    fn replay_interpolation_migration_preserves_explicit_method_presence() {
        for canonical in [None, Some("direct"), Some("auto"), Some("sol")] {
            let mut config = aic_config();
            config.fpm_interpolation = Some(crate::FpmInterpolationMethod::Direct);
            config.estimator_config = serde_json::from_value(serde_json::json!({
                "correction": {"enabled": false},
                "fpm_interpolation": canonical
                    .map(|method| serde_json::json!({"method": method}))
                    .unwrap_or_else(|| serde_json::json!({}))
            }))
            .unwrap();
            let request = config.estimator_request(ForwardPassWorkerType::Aggregated);
            if canonical.is_none() || canonical == Some("direct") {
                let controls = request.unwrap().estimator_config;
                assert_eq!(
                    controls.fpm_interpolation.method,
                    crate::FpmInterpolationMethod::Direct
                );
                assert!(!controls.correction.enabled);
            } else {
                assert!(request.unwrap_err().to_string().contains(
                    "conflicting explicit values for estimator_config.fpm_interpolation.method"
                ));
            }
        }
    }

    #[test]
    fn canonical_timing_config_preserves_moe_kernel_source() {
        let mut config = aic_config();
        config.moe_kernel_source = Some("sglang_flashinfer_trtllm_moe".into());

        assert_eq!(
            config
                .estimator_request(ForwardPassWorkerType::Aggregated)
                .unwrap()
                .moe_kernel_source
                .as_deref(),
            Some("sglang_flashinfer_trtllm_moe")
        );
    }

    #[test]
    fn fitted_replay_rejects_unsupported_work_before_loading_a_learned_source() {
        pyo3::prepare_freethreaded_python();
        for source in ["artifact_path", "checkpoint_path"] {
            let learned = || {
                let mut config = aic_config();
                config.estimation_mode = Some(EstimationMode::FpmLearned);
                config.estimator_config = serde_json::from_value(serde_json::json!({
                    "fpm_learned": {(source): "/missing/learned-source", "adaptation": "frozen"}
                }))
                .unwrap();
                config
            };
            let build = |mut config: AicTimingConfig, vision: bool| {
                AicTimingModel::build(&mut config, ForwardPassWorkerType::Decode, vision)
                    .err()
                    .expect("this learned replay must fail")
                    .to_string()
            };
            let mut speculative = learned();
            speculative.nextn = 1;
            assert!(build(speculative, false).contains("does not support speculative decoding"));
            assert!(build(learned(), true).contains("vision encoder"));
            let mut decoder = learned();
            decoder.decoder_replay = true;
            assert!(build(decoder, false).contains("bounded decoder replay"));
            // Supported work reaches the source, which is missing.
            assert!(build(learned(), false).contains("could not construct"));
        }
    }

    /// Wall time of a synthetic iteration, linear in its totals.
    fn observed_iteration(ranks: &[RankQuery]) -> Vec<crate::ForwardPassMetrics> {
        let total = |index: usize| ranks.iter().map(|rank| f64::from(rank[index])).sum::<f64>();
        let wall_ms = 1.5 + 0.004 * total(1) + 0.0002 * total(2) + 0.0003 * total(4);
        let mut metrics: Vec<_> = ranks.iter().map(AicTimingModel::rank_metrics).collect();
        metrics[0].wall_time = wall_ms / 1000.0;
        metrics
    }

    /// Every kind of aggregated iteration, unless `kinds` leaves some out.
    fn fitted_training_iterations(
        kinds: &[IterationKind],
        cross_rank: bool,
    ) -> Vec<Vec<crate::ForwardPassMetrics>> {
        let mut iterations = Vec::new();
        for i in 0..24_u32 {
            let prefill = [1 + i % 3, 64 * (1 + i % 7), 16 * (i % 5), 0, 0];
            let decode = [0, 0, 0, 1 + i % 8, 512 * (1 + i % 9)];
            let mixed = [
                1,
                32 * (1 + i % 5),
                8 * (i % 3),
                1 + i % 4,
                256 * (1 + i % 6),
            ];
            for (kind, ranks) in [
                (IterationKind::Prefill, vec![prefill]),
                (IterationKind::Decode, vec![decode]),
                (IterationKind::Mixed, vec![mixed]),
            ] {
                if kinds.contains(&kind) {
                    iterations.push(observed_iteration(&ranks));
                }
            }
            if cross_rank {
                iterations.push(observed_iteration(&[prefill, decode]));
            }
        }
        iterations
    }

    /// A planner-style identity: the deployment's hardware, versions and
    /// shape are unknown, its limits recorded.
    fn unknown_regression_identity(
        worker_type: ForwardPassWorkerType,
    ) -> ForwardPassPerfModelConfig {
        let mut config = ForwardPassPerfModelConfig::new(
            "Qwen/Qwen3-0.6B",
            crate::UNKNOWN_IDENTITY,
            crate::BackendKind::Vllm,
            worker_type,
        );
        config.estimation_mode = EstimationMode::FpmRegression;
        config.backend_version = Some(crate::UNKNOWN_IDENTITY.into());
        (config.tp, config.pp, config.attention_dp) = (UNKNOWN_SIZE, UNKNOWN_SIZE, UNKNOWN_SIZE);
        config.kv_block_size = Some(16);
        config.deployment_limits = Some(DeploymentLimits {
            num_gpus: crate::DeploymentLimit::Known(2),
            max_kv_tokens: crate::DeploymentLimit::Known(4096),
            ..Default::default()
        });
        config
    }

    /// [`unknown_regression_identity`] calibrated on an SGLang deployment.
    fn unknown_sglang_identity(worker_type: ForwardPassWorkerType) -> ForwardPassPerfModelConfig {
        let mut config = unknown_regression_identity(worker_type);
        config.backend = crate::BackendKind::Sglang;
        config
    }

    /// Save `model` and return the timing configuration replay restores it with.
    fn snapshot_timing_config(
        model: &ForwardPassPerfModel,
        path: &std::path::Path,
    ) -> AicTimingConfig {
        let saved = model
            .save_state_snapshot(path, &crate::StateSnapshotOptions::default())
            .unwrap();
        serde_json::from_value(serde_json::to_value(saved).unwrap()).unwrap()
    }

    /// A rank's work: prefill `(new_tokens, prefix_tokens)` requests, each
    /// completing its prompt in this pass, and decode context lengths.
    fn fitted_work(
        dp_rank: u32,
        prefill: Option<&[(usize, usize)]>,
        decode: Option<&[usize]>,
    ) -> RankWork {
        RankWork {
            dp_rank,
            prefill: prefill.map(|requests| crate::engine::PrefillWork {
                totals: PrefillTotals {
                    batch_size: requests.len(),
                    new_tokens: requests.iter().map(|request| request.0).sum(),
                    prefix_tokens: requests.iter().map(|request| request.1).sum(),
                },
                requests: requests
                    .iter()
                    .map(|&(new_tokens, prefix_tokens)| {
                        (new_tokens, prefix_tokens, new_tokens + prefix_tokens)
                    })
                    .collect(),
            }),
            decode: decode.map(|lengths| {
                let active_kv_tokens = lengths.iter().sum::<usize>();
                crate::engine::DecodeWork {
                    batch_size: lengths.len(),
                    active_kv_tokens,
                    mean_context_length: active_kv_tokens / lengths.len(),
                    // A clamp below the logical total that fitted queries ignore.
                    kv_bound_tokens: active_kv_tokens / 2,
                    lengths: lengths.to_vec(),
                    verify_width: 1,
                }
            }),
        }
    }

    /// The telemetry totals of `rank`, as a hand-written expectation: decode
    /// KV is past KV, each context length but its current token, as Dynamo's
    /// vLLM telemetry sums decode requests' computed tokens.
    fn telemetry_query(rank: &RankWork) -> RankQuery {
        let prefill = rank.prefill.as_ref().map_or([0; 3], |prefill| {
            [
                prefill.totals.batch_size,
                prefill.totals.new_tokens,
                prefill.totals.prefix_tokens,
            ]
            .map(|value| value as u32)
        });
        let decode = rank.decode.as_ref().map_or([0; 2], |decode| {
            [
                decode.batch_size as u32,
                decode
                    .lengths
                    .iter()
                    .map(|length| length - 1)
                    .sum::<usize>() as u32,
            ]
        });
        [prefill[0], prefill[1], prefill[2], decode[0], decode[1]]
    }

    #[test]
    fn fitted_regression_prices_each_iteration_as_the_canonical_estimate() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let identity = unknown_regression_identity(ForwardPassWorkerType::Aggregated);
        let mut model = ForwardPassPerfModel::best_available(identity).unwrap();
        model
            .tune_with_fpms(&fitted_training_iterations(
                &[
                    IterationKind::Prefill,
                    IterationKind::Decode,
                    IterationKind::Mixed,
                ],
                true,
            ))
            .unwrap();
        assert!(
            model
                .readiness_report()
                .stores
                .iter()
                .all(|store| store.ready)
        );
        let mut config = snapshot_timing_config(&model, &dir.path().join("snapshot"));
        assert_eq!(
            (config.tp, config.attention_dp),
            (UNKNOWN_SIZE, UNKNOWN_SIZE)
        );
        let timing =
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false).unwrap();
        assert!(timing.is_fitted());
        assert_eq!(
            timing.iteration.as_ref().unwrap().query,
            IterationQuery::Fitted { detail: false }
        );

        let prefill = fitted_work(0, Some(&[(200, 40), (100, 0)]), None);
        let decode = fitted_work(0, None, Some(&[700, 900, 1100]));
        let mixed = fitted_work(0, Some(&[(96, 16)]), Some(&[300, 500]));
        let decode_one = fitted_work(1, None, Some(&[400, 1600]));
        let prefill_one = fitted_work(1, Some(&[(128, 64)]), None);
        for ranks in [
            // dp=1
            vec![prefill.clone()],
            vec![decode.clone()],
            vec![mixed.clone()],
            // dp=2, idle ranks included
            vec![prefill.clone(), decode_one.clone()],
            vec![decode.clone(), decode_one.clone()],
            vec![mixed.clone(), prefill_one.clone()],
            vec![prefill.clone(), RankWork::idle(1)],
        ] {
            let queries: Vec<_> = ranks.iter().map(telemetry_query).collect();
            let expected = canonical_ms(&model, &queries);
            assert_eq!(iteration_ms(&timing, &ranks), expected, "{queries:?}");
            // Priced again from the cache.
            assert_eq!(iteration_ms(&timing, &ranks), expected);
        }
        // A group without model work takes no time and asks the model
        // nothing (see `fitted_role_passes_without_model_work_take_no_time`).
        assert_eq!(
            timing
                .predict_iteration_ms(&[RankWork::idle(0), RankWork::idle(1)])
                .unwrap(),
            Some(IterationMs { total_ms: 0.0 })
        );
        // Latency-only, and never priced by phase.
        assert!(timing.evidence_summary().is_none());
        assert!(timing.fpm_evidence_summary().is_none());
        assert!(timing.predict_decode_ms(1, 100, 100, 1000).is_err());
        assert!(timing.predict_prefill_totals_ms(1, 100, 0).is_err());
        assert!(
            timing
                .predict_mixed_pass_ms(
                    PrefillTotals {
                        batch_size: 1,
                        new_tokens: 8,
                        prefix_tokens: 0
                    },
                    1,
                    10,
                    10,
                    100
                )
                .is_err()
        );
    }

    #[test]
    fn fitted_role_passes_without_model_work_take_no_time() {
        use crate::engine::generalized::{EngineIdentity, SchedulerCommand};
        use crate::engine::{Command, EngineConfig, EngineFactory, Request, SglangConfig};
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let mut model = ForwardPassPerfModel::best_available(unknown_sglang_identity(
            ForwardPassWorkerType::Aggregated,
        ))
        .unwrap();
        // An SGLang rank never computes prefill and decode together, so its
        // telemetry has no locally mixed pass and the role needs none.
        model
            .tune_with_fpms(&fitted_training_iterations(
                &[IterationKind::Prefill, IterationKind::Decode],
                true,
            ))
            .unwrap();
        let mut config = snapshot_timing_config(&model, &dir.path().join("snapshot"));
        let timing = Arc::new(
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false).unwrap(),
        );
        assert!(!timing.prices_phases());
        // SGLang with a prefill-decode interval of 2 and attention DP 2: one
        // 12-token prompt on rank 0 runs in 4-token chunks, each followed by
        // two deferred rounds on which neither rank computes model work.
        let engine_config = EngineConfig {
            num_gpu_blocks: 1024,
            block_size: 1,
            max_num_seqs: 16,
            max_num_batched_tokens: 128,
            prefill_decode_interval: 2,
            enable_prefix_caching: false,
            sglang: SglangConfig {
                // Divided among the two ranks.
                chunked_prefill_size: 8,
                schedule_conservativeness: 0.0,
                ..SglangConfig::default()
            },
            ..EngineConfig::for_backend(Backend::Sglang)
        };
        let mut engine = EngineFactory::with_timing_model(engine_config, timing)
            .unwrap()
            .build(
                EngineIdentity::new(0),
                std::num::NonZeroU32::new(2).unwrap(),
            )
            .unwrap();
        engine
            .apply_command_effects(
                SchedulerCommand::new(
                    0,
                    Command::Submit(Request {
                        request_id: uuid::Uuid::from_u128(1),
                        tokens: vec![1; 12],
                        max_output_tokens: 1,
                        output_token_ids: None,
                        session_id: None,
                        images: Vec::new(),
                        kv_hint: None,
                    }),
                ),
                0.0,
            )
            .unwrap();
        let mut now_ms = 0.0;
        let mut rounds = Vec::new();
        for _ in 0..7 {
            // A phase query would fail: the fitted role has no phase pricing.
            let started = engine.execute_pass(now_ms).unwrap().unwrap();
            let completed = engine
                .complete_pass(started.pass_id, started.end_ms)
                .unwrap();
            let fpm = &completed.effects.by_rank[0].effects.forward_pass_metrics;
            rounds.push((
                fpm.num_prefill_requests,
                started.end_ms - started.started_at_ms,
            ));
            now_ms = started.end_ms;
        }
        for (round, (prefills, duration_ms)) in rounds.iter().enumerate() {
            if round % 3 == 0 {
                assert_eq!(*prefills, 1, "{rounds:?}");
                assert!(*duration_ms > 0.0, "{rounds:?}");
            } else {
                assert_eq!((*prefills, *duration_ms), (0, 0.0), "{rounds:?}");
            }
        }
    }

    #[test]
    fn fitted_replay_rejects_a_decode_speedup_other_than_one() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let mut model = ForwardPassPerfModel::best_available(unknown_regression_identity(
            ForwardPassWorkerType::Decode,
        ))
        .unwrap();
        model
            .tune_with_fpms(&fitted_training_iterations(&[IterationKind::Decode], false))
            .unwrap();
        let saved = model
            .save_state_snapshot(
                &dir.path().join("snapshot"),
                &crate::StateSnapshotOptions::default(),
            )
            .unwrap();
        let resolve = |decode_speedup_ratio: f64| {
            let mut role = ReplayRoleConfig::default();
            role.rank.block_size = 16;
            role.rank.decode_speedup_ratio = decode_speedup_ratio;
            role.rank.timing_model = TimingModelConfig::External {
                provider: "aic".into(),
                config: serde_json::to_value(&saved).unwrap(),
            };
            resolve_role_timing(
                &mut role,
                false,
                ForwardPassWorkerType::Decode,
                false,
                &mut Vec::new(),
            )
            .map(|timing| (timing.is_some(), role.rank.num_gpu_blocks))
        };
        // At a unit decode speedup the role takes its recorded 4096 KV
        // tokens in 16-token blocks.
        assert_eq!(resolve(1.0).unwrap(), (true, 256));
        for ratio in [0.5, 2.0] {
            let error = resolve(ratio).unwrap_err().to_string();
            assert!(
                error.contains("fpm_regression replay requires decode_speedup_ratio=1.0"),
                "{error}"
            );
        }
    }

    #[test]
    fn fitted_replay_requires_every_store_its_role_prices() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let build = |kinds: &[IterationKind],
                     cross_rank: bool,
                     worker: ForwardPassWorkerType,
                     name: &str| {
            let mut model =
                ForwardPassPerfModel::best_available(unknown_regression_identity(worker)).unwrap();
            let iterations = fitted_training_iterations(kinds, cross_rank);
            if !iterations.is_empty() {
                model.tune_with_fpms(&iterations).unwrap();
            }
            let mut config = snapshot_timing_config(&model, &dir.path().join(name));
            (model, AicTimingModel::build(&mut config, worker, false))
        };
        let message =
            |result: Result<AicTimingModel>| result.err().expect("unready stores").to_string();
        // A cold regression has no store.
        let (_, cold) = build(&[], false, ForwardPassWorkerType::Decode, "cold");
        assert!(
            message(cold).contains("workload store(s) pure_decode (insufficient_data) not ready")
        );
        // An aggregated worker also needs prefill and locally mixed stores.
        let (_, decode_only) = build(
            &[IterationKind::Decode],
            false,
            ForwardPassWorkerType::Aggregated,
            "decode-only",
        );
        let error = message(decode_only);
        assert!(
            error.contains(
                "contains_locally_mixed (insufficient_data), pure_prefill (insufficient_data)"
            ) && !error.contains("pure_decode"),
            "{error}"
        );
        // Without cross-rank observations the role builds for one rank only.
        let (model, timing) = build(
            &[
                IterationKind::Prefill,
                IterationKind::Decode,
                IterationKind::Mixed,
            ],
            false,
            ForwardPassWorkerType::Aggregated,
            "local",
        );
        timing.unwrap();
        require_iteration_stores(&model, ForwardPassWorkerType::Aggregated, 1).unwrap();
        let error = require_iteration_stores(&model, ForwardPassWorkerType::Aggregated, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cross_rank_aggregated"), "{error}");
        // A dedicated role needs only its own store.
        let (_, decode) = build(
            &[IterationKind::Decode],
            false,
            ForwardPassWorkerType::Decode,
            "decode",
        );
        decode.unwrap();
        // Speculation and vision are rejected for a restored regression too.
        let mut model = ForwardPassPerfModel::best_available(unknown_regression_identity(
            ForwardPassWorkerType::Decode,
        ))
        .unwrap();
        model
            .tune_with_fpms(&fitted_training_iterations(&[IterationKind::Decode], false))
            .unwrap();
        let config = snapshot_timing_config(&model, &dir.path().join("rejections"));
        let mut speculative = config.clone();
        speculative.nextn = 1;
        assert!(
            message(AicTimingModel::build(
                &mut speculative,
                ForwardPassWorkerType::Decode,
                false
            ))
            .contains("does not support speculative decoding")
        );
        let mut vision = config;
        assert!(
            message(AicTimingModel::build(
                &mut vision,
                ForwardPassWorkerType::Decode,
                true
            ))
            .contains("vision encoder")
        );
    }

    #[test]
    fn fitted_rank_metrics_carry_telemetry_totals_and_request_detail() {
        let mixed = fitted_work(3, Some(&[(200, 40), (100, 0)]), Some(&[700, 900, 1100]));
        let query = AicTimingModel::fitted_rank_query(&mixed).unwrap();
        // Decode KV is past KV, as Dynamo's vLLM telemetry sums each decode
        // request's computed tokens: contexts 700, 900 and 1100 each hold one
        // current token, so 699 + 899 + 1099 = 2697, unclamped by the
        // scheduler's KV bound of 1350.
        assert_eq!(query, [2, 300, 40, 3, 2697]);
        let metrics = AicTimingModel::detailed_rank_metrics(&query, &mixed).unwrap();
        let scheduled = &metrics.scheduled_requests;
        assert_eq!(
            scheduled.extend_lengths.as_deref(),
            Some([200, 100, 1, 1, 1].as_slice())
        );
        assert_eq!(
            scheduled.past_kv_lengths.as_deref(),
            Some([40, 0, 699, 899, 1099].as_slice())
        );
        // The decode requests' past KV sums to the totals' decode KV.
        let decode_past: u64 = scheduled.past_kv_lengths.as_ref().unwrap()[2..]
            .iter()
            .sum();
        assert_eq!(decode_past, u64::from(query[4]));
        // Prefill lengths 240 and 100: mean 170, variance 70^2. Decode past
        // KV 699, 899 and 1099: mean 899, variance 2 * 200^2 / 3.
        assert_eq!(scheduled.var_prefill_length, 4900.0);
        assert!((scheduled.var_decode_kv_tokens - 80_000.0 / 3.0).abs() < 1e-9);
        crate::perfmodel::fpm::validate_forward_pass_metrics(&metrics).unwrap();
        // A first decode step after a 1-token prompt has no past KV.
        let first = fitted_work(0, None, Some(&[1]));
        assert_eq!(
            AicTimingModel::fitted_rank_query(&first).unwrap(),
            [0, 0, 0, 1, 0]
        );
        let idle = AicTimingModel::detailed_rank_metrics(&[0; 5], &RankWork::idle(0)).unwrap();
        assert_eq!(idle.scheduled_requests.extend_lengths, None);
        // Scheduler lists that disagree with the totals are an error.
        let mut short = mixed.clone();
        short.decode.as_mut().unwrap().lengths.pop();
        assert!(AicTimingModel::detailed_rank_metrics(&query, &short).is_err());
        let mut speculative = mixed.clone();
        speculative.decode.as_mut().unwrap().verify_width = 2;
        assert!(AicTimingModel::fitted_rank_query(&speculative).is_err());
        // Decode totals without a current token per request are an error.
        let mut no_current = mixed;
        no_current.decode.as_mut().unwrap().active_kv_tokens = 2;
        assert!(AicTimingModel::fitted_rank_query(&no_current).is_err());
    }

    #[test]
    fn fitted_regression_with_request_axes_reads_per_request_detail() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let mut identity = unknown_regression_identity(ForwardPassWorkerType::Decode);
        identity.estimator_config = serde_json::from_value(serde_json::json!({
            "fpm_regression": {"fit": {"linear": {"feature_axes": ["attention", "moe", "P2"]}}}
        }))
        .unwrap();
        let mut model = ForwardPassPerfModel::best_available(identity).unwrap();
        let ranks = |i: usize| {
            let lengths: Vec<usize> = (0..1 + i % 5)
                .map(|j| 100 + 37 * ((i * 7 + j * 13) % 41))
                .collect();
            fitted_work(0, None, Some(&lengths))
        };
        let iterations: Vec<_> = (0..32)
            .map(|i| {
                let rank = ranks(i);
                let query = AicTimingModel::fitted_rank_query(&rank).unwrap();
                let mut metrics = AicTimingModel::detailed_rank_metrics(&query, &rank).unwrap();
                let squares: f64 = metrics
                    .scheduled_requests
                    .past_kv_lengths
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|p| (*p as f64).powi(2))
                    .sum();
                metrics.wall_time = (2.0 + 0.001 * f64::from(query[4]) + 1e-6 * squares) / 1000.0;
                vec![metrics]
            })
            .collect();
        model.tune_with_fpms(&iterations).unwrap();
        assert!(model.reads_request_detail());
        let mut config = snapshot_timing_config(&model, &dir.path().join("snapshot"));
        let timing =
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Decode, false).unwrap();
        let pricing = timing.iteration.as_ref().unwrap();
        assert_eq!(pricing.query, IterationQuery::Fitted { detail: true });
        assert!(pricing.cache.is_none());
        for i in [3, 8, 21] {
            let rank = ranks(i);
            let query = AicTimingModel::fitted_rank_query(&rank).unwrap();
            let expected = model
                .estimate_forward_pass_time_ms(&[AicTimingModel::detailed_rank_metrics(
                    &query, &rank,
                )
                .unwrap()])
                .unwrap()
                .unwrap();
            assert_eq!(iteration_ms(&timing, &[rank]), expected);
        }
    }

    #[test]
    fn fitted_aggregated_role_needs_a_locally_mixed_store_only_where_ranks_mix_work() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        for backend in [
            crate::BackendKind::Vllm,
            crate::BackendKind::Trtllm,
            crate::BackendKind::Sglang,
        ] {
            let mut identity = unknown_regression_identity(ForwardPassWorkerType::Aggregated);
            identity.backend = backend.clone();
            let mut model = ForwardPassPerfModel::best_available(identity).unwrap();
            // Pure prefill and pure decode telemetry only.
            model
                .tune_with_fpms(&fitted_training_iterations(
                    &[IterationKind::Prefill, IterationKind::Decode],
                    false,
                ))
                .unwrap();
            let mut config = snapshot_timing_config(&model, &dir.path().join(backend.as_str()));
            let built =
                AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false);
            if backend == crate::BackendKind::Sglang {
                // An SGLang rank runs a prefill batch alone, so no pass of
                // the role is locally mixed.
                built.unwrap();
                require_iteration_stores(&model, ForwardPassWorkerType::Aggregated, 1).unwrap();
                // Its attention-DP ranks still compute different work.
                let error = require_iteration_stores(&model, ForwardPassWorkerType::Aggregated, 2)
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains("cross_rank_aggregated")
                        && !error.contains("contains_locally_mixed"),
                    "{error}"
                );
            } else {
                // vLLM and TensorRT-LLM schedule chunked prefill beside decode.
                let error = built.err().expect("no locally mixed store").to_string();
                assert!(
                    error.contains(
                        "workload store(s) contains_locally_mixed (insufficient_data) not ready"
                    ),
                    "{backend:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn fitted_aggregated_role_checks_its_cross_rank_store_at_the_engine_width() {
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        // The identity records unknown attention DP and two GPUs; engine
        // dp_size 2 at tensor parallelism 1 uses both.
        let resolve = |cross_rank: bool, dp_size: u32| {
            let mut model = ForwardPassPerfModel::best_available(unknown_regression_identity(
                ForwardPassWorkerType::Aggregated,
            ))
            .unwrap();
            model
                .tune_with_fpms(&fitted_training_iterations(
                    &[
                        IterationKind::Prefill,
                        IterationKind::Decode,
                        IterationKind::Mixed,
                    ],
                    cross_rank,
                ))
                .unwrap();
            let saved = model
                .save_state_snapshot(
                    &dir.path().join(format!("{cross_rank}-{dp_size}")),
                    &crate::StateSnapshotOptions::default(),
                )
                .unwrap();
            let mut role = ReplayRoleConfig {
                dp_size,
                ..Default::default()
            };
            role.rank.block_size = 16;
            role.rank.timing_model = TimingModelConfig::External {
                provider: "aic".into(),
                config: serde_json::to_value(&saved).unwrap(),
            };
            resolve_role_timing(
                &mut role,
                false,
                ForwardPassWorkerType::Aggregated,
                false,
                &mut Vec::new(),
            )
            .map(|timing| timing.is_some())
        };
        assert!(resolve(false, 1).unwrap());
        // Construction cannot see the engine's width; the role check does.
        let error = resolve(false, 2).unwrap_err().to_string();
        assert!(
            error.contains("workload store(s) cross_rank_aggregated")
                && error.contains("not ready"),
            "{error}"
        );
        assert!(resolve(true, 2).unwrap());
    }

    #[test]
    fn fitted_kvcr_role_with_an_unknown_pipeline_width_runs_one_stage() {
        use crate::engine::{KvcrConfig, NativeHostOffloadConfig};
        use crate::replay::WorkerStage;
        pyo3::prepare_freethreaded_python();
        let dir = tempfile::tempdir().unwrap();
        let mut model = ForwardPassPerfModel::best_available(unknown_regression_identity(
            ForwardPassWorkerType::Aggregated,
        ))
        .unwrap();
        model
            .tune_with_fpms(&fitted_training_iterations(
                &[
                    IterationKind::Prefill,
                    IterationKind::Decode,
                    IterationKind::Mixed,
                ],
                false,
            ))
            .unwrap();
        let saved = model
            .save_state_snapshot(
                &dir.path().join("snapshot"),
                &crate::StateSnapshotOptions::default(),
            )
            .unwrap();
        let mut role = ReplayRoleConfig::default();
        role.rank.block_size = 16;
        role.rank.kv_cache_bytes_per_token = Some(250_000);
        role.rank.native_host_offload = Some(NativeHostOffloadConfig::new(4));
        role.rank.kvcr = Some(KvcrConfig::default());
        role.rank.timing_model = TimingModelConfig::External {
            provider: "aic".into(),
            config: serde_json::to_value(&saved).unwrap(),
        };
        let timing = resolve_role_timing(
            &mut role,
            false,
            ForwardPassWorkerType::Aggregated,
            false,
            &mut Vec::new(),
        )
        .unwrap()
        .unwrap();
        // The resolved identity the role keeps spells its pipeline width unknown.
        let TimingModelConfig::External { config, .. } = &role.rank.timing_model else {
            panic!("the role keeps its AIC timing");
        };
        assert_eq!(config["pp"], "unknown");
        let engine = ReplayEngineConfig {
            rank: role.rank.clone(),
            ..Default::default()
        };
        let factory = ReplayEngineFactory::with_timing_model(timing)
            .role_factory(&engine, WorkerStage::Aggregated, false)
            .unwrap();
        assert_eq!(factory.kvcr_config, Some(KvcrConfig::default()));
    }

    #[test]
    fn fitted_metrics_number_every_rank_of_the_group() {
        let ranks = [
            fitted_work(0, Some(&[(64, 0)]), None),
            RankWork::idle(1),
            fitted_work(2, None, Some(&[10, 20])),
        ];
        let queries: Vec<_> = ranks
            .iter()
            .map(|rank| AicTimingModel::fitted_rank_query(rank).unwrap())
            .collect();
        for detail in [false, true] {
            let metrics = AicTimingModel::fitted_metrics(&queries, &ranks, detail).unwrap();
            // A learned model's mixed composition requires the complete,
            // unique rank IDs 0..attention_dp.
            assert_eq!(
                metrics
                    .iter()
                    .map(|metrics| metrics.dp_rank)
                    .collect::<Vec<_>>(),
                [0, 1, 2]
            );
            assert_eq!(metrics[1].scheduled_requests, Default::default());
        }
    }

    /// Records each group pass's work and prices it at 1 ms.
    #[derive(Default)]
    struct CapturedIterations(Mutex<Vec<Vec<RankWork>>>);

    impl TimingModel for CapturedIterations {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> Result<f64> {
            anyhow::bail!("passes are priced as iterations")
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> Result<f64> {
            anyhow::bail!("passes are priced as iterations")
        }
        fn predict_iteration_ms(&self, ranks: &[RankWork]) -> Result<Option<IterationMs>> {
            self.0.lock().unwrap().push(ranks.to_vec());
            Ok(Some(IterationMs { total_ms: 1.0 }))
        }
        fn prices_phases(&self) -> bool {
            false
        }
    }

    #[test]
    fn fitted_detail_reports_the_full_prompt_variance_of_a_chunked_prefill() {
        use crate::engine::generalized::{EngineIdentity, SchedulerCommand};
        use crate::engine::{Command, EngineFactory, Request, SglangConfig};
        // The first 4096-token chunk of an 8000-token prompt beside a whole
        // 4000-token prompt. Dynamo's vLLM telemetry reports the prompts'
        // lengths, 8000 and 4000: mean 6000, variance 2000^2. The tokens
        // through this pass, 4096 and 4000, would give 48^2.
        let mut chunked = fitted_work(0, Some(&[(4096, 0), (4000, 0)]), None);
        chunked.prefill.as_mut().unwrap().requests[0].2 = 8000;
        let query = AicTimingModel::fitted_rank_query(&chunked).unwrap();
        let metrics = AicTimingModel::detailed_rank_metrics(&query, &chunked).unwrap();
        assert_eq!(metrics.scheduled_requests.var_prefill_length, 4_000_000.0);
        assert_eq!(
            metrics.scheduled_requests.extend_lengths.as_deref(),
            Some([4096, 4000].as_slice())
        );

        // Both schedulers carry each request's prompt length: on every pass of
        // a chunked prefill the detail's variance is the one their own
        // forward-pass metrics publish.
        for backend in [Backend::Vllm, Backend::Sglang] {
            let timing = Arc::new(CapturedIterations::default());
            let engine_config = EngineConfig {
                num_gpu_blocks: 1024,
                block_size: 16,
                max_num_seqs: 16,
                max_num_batched_tokens: 64,
                enable_chunked_prefill: true,
                enable_prefix_caching: false,
                sglang: SglangConfig {
                    chunked_prefill_size: 64,
                    ..SglangConfig::default()
                },
                ..EngineConfig::for_backend(backend)
            };
            let mut engine = EngineFactory::with_timing_model(
                engine_config,
                Arc::clone(&timing) as Arc<dyn TimingModel>,
            )
            .unwrap()
            .build(EngineIdentity::new(0), std::num::NonZeroU32::MIN)
            .unwrap();
            // A short prompt first, so the long one is cut to the rest of the
            // first pass's token budget.
            for (id, length) in [(1, 40), (2, 200)] {
                engine
                    .apply_command_effects(
                        SchedulerCommand::new(
                            0,
                            Command::Submit(Request {
                                request_id: uuid::Uuid::from_u128(id),
                                tokens: (0..length).map(|token| token as u32).collect(),
                                max_output_tokens: 2,
                                output_token_ids: None,
                                session_id: None,
                                images: Vec::new(),
                                kv_hint: None,
                            }),
                        ),
                        0.0,
                    )
                    .unwrap();
            }
            let (mut now_ms, mut mid_chunk_mixed_lengths) = (0.0, 0);
            while let Some(started) = engine.execute_pass(now_ms).unwrap() {
                let completed = engine
                    .complete_pass(started.pass_id, started.end_ms)
                    .unwrap();
                now_ms = started.end_ms;
                let published = &completed.effects.by_rank[0].effects.forward_pass_metrics;
                let Some(ranks) = timing.0.lock().unwrap().pop() else {
                    continue;
                };
                let rank = &ranks[0];
                let Some(prefill) = rank.prefill.as_ref() else {
                    continue;
                };
                let query = AicTimingModel::fitted_rank_query(rank).unwrap();
                let detail = AicTimingModel::detailed_rank_metrics(&query, rank).unwrap();
                assert!(
                    (detail.scheduled_requests.var_prefill_length - published.var_prefill_length)
                        .abs()
                        < 1e-6,
                    "{backend:?}: {prefill:?} vs {published:?}"
                );
                if prefill.requests.len() > 1
                    && prefill
                        .requests
                        .iter()
                        .any(|&(new_tokens, prefix, prompt)| new_tokens + prefix < prompt)
                {
                    mid_chunk_mixed_lengths += 1;
                }
            }
            assert!(
                mid_chunk_mixed_lengths > 0,
                "{backend:?} never cut a prompt beside another prefill"
            );
        }
    }

    fn aic_config() -> AicTimingConfig {
        AicTimingConfig {
            model: "test-model".into(),
            backend: "vllm".into(),
            system: "test-system".into(),
            tp: 1,
            backend_version: None,
            pp: 1,
            attention_dp: 1,
            dcp: None,
            moe_tp_size: None,
            moe_ep_size: None,
            cp_size: None,
            gemm_dtype: None,
            moe_dtype: None,
            fmha_dtype: None,
            fpm_fmha_dtype: None,
            kv_cache_dtype: None,
            comm_dtype: None,
            nextn: 0,
            speculation: None,
            kv_block_size: None,
            gpu_memory_utilization: None,
            mem_fraction_static: None,
            free_gpu_memory_fraction: None,
            cuda_graph_reserved_bytes: 0,
            worker_type: None,
            estimation_mode: None,
            fallback_policy: ForwardPassFallbackPolicy::Deny,
            estimator_config: serde_json::Map::new(),
            database_mode: crate::DatabaseMode::default(),
            transfer_policy: None,
            systems_paths: Vec::new(),
            attention_backend: None,
            moe_backend: None,
            enable_eplb: false,
            wideep_num_slots: None,
            moe_kernel_source: None,
            enable_shared_layer: None,
            strict_provenance: false,
            systems_path: None,
            forward_model: None,
            fpm_profile: None,
            fpm_interpolation: None,
            fpm_parquet_path: None,
            decoder_replay: false,
            encoder_parallel: None,
            deployment_limits: None,
        }
    }

    #[test]
    fn parallel_shape_folds_prefill_cp_but_not_decode_cp_into_width() {
        let mut config = aic_config();
        config.tp = 1;
        config.attention_dp = 1;
        config.moe_tp_size = Some(1);
        config.moe_ep_size = Some(8);
        // Prefill CP widens the attention side to match the MoE width ...
        config.cp_size = Some(8);
        config.dcp = Some(8);
        config.validate_parallel_shape().unwrap();
        // ... decode CP does not: without prefill CP the widths no longer match.
        config.cp_size = None;
        assert!(config.validate_parallel_shape().is_err());
        // Zero is rejected like every other parallel size.
        config.cp_size = Some(8);
        config.dcp = Some(0);
        assert!(config.validate_parallel_shape().is_err());
    }

    #[test]
    fn fixed_capacity_does_not_require_vllm_pp1_profile_resources() {
        for (backend, pp) in [("sglang", 1), ("vllm", 2)] {
            let mut config = aic_config();
            config.backend = backend.into();
            config.pp = pp;
            config.fpm_profile = Some(serde_json::Map::new());
            let mut role = aggregated_role(&ReplayEngineConfig::default());
            role.rank.backend = if backend == "sglang" {
                Backend::Sglang
            } else {
                Backend::Vllm
            };
            materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();
        }
    }

    #[test]
    fn measured_profile_capacity_forwards_depth_without_changing_analytical_defaults() {
        pyo3::prepare_freethreaded_python();
        let role = aggregated_role(&ReplayEngineConfig::default());
        let mut config = aic_config();
        Python::with_gil(|py| {
            for depth in [0, 3] {
                config.nextn = depth;
                let kwargs = aic_capacity_kwargs(py, &config, &role).unwrap();
                assert!(!kwargs.contains("nextn").unwrap());
            }
            // Selection/resource validity is enforced by materialize_aic_capacity.
            // This unit test isolates only the previously missing kwargs hop.
            config.fpm_profile = Some(serde_json::Map::new());
            let kwargs = aic_capacity_kwargs(py, &config, &role).unwrap();
            assert_eq!(
                kwargs
                    .get_item("nextn")
                    .unwrap()
                    .unwrap()
                    .extract::<u32>()
                    .unwrap(),
                3
            );
            config.nextn = 0;
            assert!(
                !aic_capacity_kwargs(py, &config, &role)
                    .unwrap()
                    .contains("nextn")
                    .unwrap()
            );
        });
    }

    #[test]
    fn grouped_load_and_timing_domain_preserve_byte_capacity() {
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        role.dp_size = 3;
        role.rank.kv_cache_groups = serde_json::from_value(serde_json::json!([{
            "name":"window", "kind":"attention", "num_layers":1,
            "block_size_tokens":16, "page_size_bytes":160, "sliding_window":32
        }]))
        .unwrap();
        role.rank.kv_cache_capacity_bytes = Some(1000);
        cap_role_capacity_to_fpm_decode_domain(&mut role, Some(1), false).unwrap();
        assert_eq!(role.rank.kv_cache_capacity_bytes, Some(1000));
        let mut traffic: RuntimeTraffic = serde_json::from_value(serde_json::json!({
            "source_type":"synthetic", "load_type":"kv_capacity_fraction",
            "kv_load_ratio":1.0, "isl":1152, "osl":4
        }))
        .unwrap();
        // A 32-token window can straddle three 16-token pages (480 bytes).
        // Each rank holds two requests; three DP ranks and two replicas hold 12.
        assert_eq!(
            resolve_kv_capacity_concurrency(&mut traffic, &role, 2).unwrap(),
            Some(12)
        );
        assert_eq!(traffic.concurrency, Some(12));
        role.rank.kv_cache_capacity_bytes = Some(479);
        assert!(resolve_kv_capacity_concurrency(&mut traffic, &role, 2).is_err());
    }

    #[test]
    fn ngram_timing_requires_matching_scheduler_depth_and_no_mtp() {
        let mut config = aic_config();
        config.speculation = Some(
            serde_json::from_value(serde_json::json!({
                "kind": "ngram", "params": {"num_speculative_tokens": 2}
            }))
            .unwrap(),
        );
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        assert!(materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).is_err());
        role.rank.aic_nextn = Some(2);
        materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();
        config.nextn = 2;
        assert!(config.validate_parallel_shape().is_err());
        config.nextn = 0;
        config.speculation = Some(
            serde_json::from_value(serde_json::json!({
                "kind": "ngram", "params": {"num_speculative_tokens": 6}
            }))
            .unwrap(),
        );
        assert!(config.validate_parallel_shape().is_err());
    }

    #[test]
    fn ngram_drafts_turn_vllm_async_scheduling_off() {
        let mut rank = EngineConfig::default();
        resolve_ngram_async_scheduling(&mut rank, "mtp").unwrap();
        assert_eq!(rank.async_scheduling, None);
        resolve_ngram_async_scheduling(&mut rank, "ngram").unwrap();
        assert_eq!(rank.async_scheduling, Some(false));
        rank.async_scheduling = Some(true);
        let error = resolve_ngram_async_scheduling(&mut rank, "ngram").unwrap_err();
        assert!(error.to_string().contains("async scheduling"), "{error}");
        let mut sglang = EngineConfig::for_backend(Backend::Sglang);
        resolve_ngram_async_scheduling(&mut sglang, "ngram").unwrap();
        assert_eq!(sglang.async_scheduling, None);
    }

    #[test]
    fn replay_rejects_unsupported_depth_and_unmodeled_ngram_parameters() {
        for payload in [
            serde_json::json!({"kind": "mtp", "params": {"depth": 6}}),
            serde_json::json!({"kind": "ngram", "params": {"num_speculative_tokens": 2, "trigger_rate": 0.5}}),
        ] {
            let mut config = aic_config();
            config.speculation = Some(serde_json::from_value(payload).unwrap());
            assert!(config.validate_parallel_shape().is_err());
        }
    }

    #[test]
    fn replay_capacity_enforces_scheme_backend_scope_without_restricting_legacy() {
        for (backend, native_backend) in [
            ("vllm", Backend::Vllm),
            ("sglang", Backend::Sglang),
            ("trtllm", Backend::Trtllm),
        ] {
            let mut config = aic_config();
            config.backend = backend.into();
            config.speculation = Some(
                serde_json::from_value(serde_json::json!({
                    "kind": "mtp", "params": {"depth": 2}
                }))
                .unwrap(),
            );
            let mut role = aggregated_role(&ReplayEngineConfig::default());
            role.rank.backend = native_backend;
            role.rank.aic_nextn = Some(2);
            let generic = materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!());
            if backend == "trtllm" {
                assert!(
                    generic
                        .unwrap_err()
                        .to_string()
                        .contains("mtp speculative replay requires backend=vllm or backend=sglang")
                );
            } else {
                generic.unwrap();
            }

            config.speculation = Some(
                serde_json::from_value(serde_json::json!({
                    "kind": "ngram", "params": {"num_speculative_tokens": 2}
                }))
                .unwrap(),
            );
            let ngram = materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!());
            if backend == "vllm" {
                ngram.unwrap();
            } else {
                assert!(
                    ngram
                        .unwrap_err()
                        .to_string()
                        .contains("ngram speculation requires backend=vllm")
                );
            }

            config.speculation = None;
            config.nextn = 2;
            materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();

            config.speculation =
                Some(serde_json::from_value(serde_json::json!({"kind": "none"})).unwrap());
            config.nextn = 0;
            role.rank.aic_nextn = None;
            materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();
        }
    }

    #[test]
    fn capacity_is_estimated_only_when_not_explicit() {
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        materialize_aic_capacity(&aic_config(), &mut role, false, |_config, rank| {
            assert_eq!(rank.rank.block_size, EngineConfig::default().block_size);
            Ok(321)
        })
        .unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 321);

        role.rank.num_gpu_blocks = 17;
        materialize_aic_capacity(
            &aic_config(),
            &mut role,
            true,
            |_config, _rank| -> Result<usize> {
                panic!("explicit capacity must not invoke the estimator")
            },
        )
        .unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 17);
        let mut config = aic_config();
        config.speculation = Some(
            serde_json::from_value(serde_json::json!({"kind": "mtp", "params": {"depth": 3}}))
                .unwrap(),
        );
        assert!(
            materialize_aic_capacity(&config, &mut role, false, |_, _| unreachable!())
                .unwrap_err()
                .to_string()
                .contains("explicit num_gpu_blocks")
        );
        role.rank.aic_nextn = Some(3);
        materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();
        config.speculation =
            Some(serde_json::from_value(serde_json::json!({"kind": "none"})).unwrap());
        role.rank.aic_nextn = Some(2);
        assert!(materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).is_err());
    }

    #[test]
    fn dcp_capacity_is_estimated_on_the_kv_stripe() {
        // The KV-capacity estimator prices the 1/dcp stripe (memory.py's
        // `_cp_kv_memory_divisor`), so automatic sizing stays available under
        // DCP; explicit capacity still bypasses the estimator.
        let mut config = aic_config();
        config.tp = 4;
        config.dcp = Some(4);
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        role.tensor_parallel_size = 4;
        role.rank.num_gpu_blocks = 17;

        materialize_aic_capacity(&config, &mut role, false, |_, _| Ok(4 * 321)).unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 4 * 321);

        role.rank.num_gpu_blocks = 17;
        materialize_aic_capacity(&config, &mut role, true, |_, _| {
            panic!("explicit DCP capacity must not invoke the estimator")
        })
        .unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 17);
    }

    #[test]
    fn cuda_graph_reservation_reaches_capacity_rematerialization() {
        let mut config = aic_config();
        config.cuda_graph_reserved_bytes = 14_559_939_133;
        let mut role = aggregated_role(&ReplayEngineConfig::default());

        materialize_aic_capacity(&config, &mut role, false, |config, _role| {
            assert_eq!(config.cuda_graph_reserved_bytes, 14_559_939_133);
            Ok(321)
        })
        .unwrap();

        assert_eq!(role.rank.num_gpu_blocks, 321);
    }

    #[test]
    fn inferred_capacity_is_capped_to_the_fpm_decode_domain() {
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        role.rank.block_size = 16;
        role.rank.num_gpu_blocks = 34_483;

        cap_role_capacity_to_fpm_decode_domain(&mut role, Some(546_046), false).unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 34_127);

        role.rank.num_gpu_blocks = 17;
        cap_role_capacity_to_fpm_decode_domain(&mut role, Some(546_046), false).unwrap();
        assert_eq!(
            role.rank.num_gpu_blocks, 17,
            "coverage must never grow capacity"
        );

        role.rank.num_gpu_blocks = 34_483;
        cap_role_capacity_to_fpm_decode_domain(&mut role, Some(546_046), true).unwrap();
        assert_eq!(
            role.rank.num_gpu_blocks, 34_483,
            "explicit capacity is authoritative"
        );
    }

    #[test]
    fn capacity_is_detected_independently_per_role() {
        let engine = serde_json::json!({
            "prefill": {"rank": {"num_gpu_blocks": 17}},
            "decode": {"rank": {}}
        });
        assert!(role_capacity_is_explicit(&engine, Some("prefill")));
        assert!(!role_capacity_is_explicit(&engine, Some("decode")));
    }

    #[test]
    fn manual_state_capacity_bypasses_native_estimation_and_fpm_capping() {
        let rank_json = serde_json::json!({"num_gpu_blocks":8,"block_size":64,"kv_cache_bytes_per_token":16,
            "state_cache": {"bytes_per_request":1500}});
        assert!(role_capacity_is_explicit(
            &serde_json::json!({"rank": rank_json}),
            None
        ));
        assert!(!role_capacity_is_explicit(
            &serde_json::json!({"rank": {"state_cache": null}}),
            None
        ));
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        role.rank = serde_json::from_value(rank_json).unwrap();
        // Even a stale/false outer hint must not override explicit manual geometry.
        materialize_aic_capacity(&aic_config(), &mut role, false, |_, _| {
            panic!("manual state capacity must never invoke the estimator")
        })
        .unwrap();
        cap_role_capacity_to_fpm_decode_domain(&mut role, Some(64), false).unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 8);
        assert_eq!(role.rank.block_size, 64);
    }

    #[test]
    fn invalid_memory_fraction_is_rejected() {
        let mut config = aic_config();
        config.gpu_memory_utilization = Some(1.1);
        assert!(
            config
                .resolved_memory_fraction()
                .unwrap_err()
                .to_string()
                .contains("gpu_memory_utilization")
        );
    }

    #[test]
    fn per_op_power_summary_matches_aic_coverage_semantics() {
        let summary = phase_evidence_from_entries(vec![
            ("covered".into(), 100.0, 50_000.0, "silicon".into()),
            ("missing".into(), 25.0, 0.0, "empirical".into()),
            ("no-op".into(), 0.0, 0.0, "silicon".into()),
        ])
        .unwrap();
        assert_eq!(summary.to_phase().energy_wms, Some(50_000.0));
        assert_eq!(summary.to_phase().latency_ms, 125.0);
        assert_eq!(summary.to_phase().covered_latency_ms, 100.0);
    }

    #[test]
    fn replay_power_applies_phase_speedups_before_weighting() {
        let source = TimingPowerSource {
            timing: Arc::new(PowerTiming(TimingEvidenceSummary {
                prefill: TimingPhaseEvidence {
                    energy_wms: Some(100_000.0),
                    latency_ms: 200.0,
                    covered_latency_ms: 190.0,
                    ..Default::default()
                },
                decode: TimingPhaseEvidence {
                    energy_wms: Some(150_000.0),
                    latency_ms: 300.0,
                    covered_latency_ms: 300.0,
                    ..Default::default()
                },
            })),
            prefill_speedup_ratio: 2.0,
            decode_speedup_ratio: 4.0,
        };
        let evidence = replay_timing_evidence(&[source]).unwrap().unwrap();
        let power = replay_power_stats(&evidence).unwrap();
        assert_eq!(power.power_w, Some(500.0));
        assert!((power.coverage - 170.0 / 175.0).abs() < 1e-12);
    }

    #[test]
    fn replay_power_returns_errors_for_non_finite_derived_evidence() {
        // Each input is finite and positive. Scaling or combining it can still
        // overflow; the provider boundary must return Err instead of panicking.
        for (energy, latency, speedup) in [
            (1.0, 1.0, f64::from_bits(1)),
            (f64::MAX, 1.0, 0.5),
            (1.0, f64::MAX, 0.5),
            (f64::MAX, 1.0, 1.0),
        ] {
            let phase = TimingPhaseEvidence {
                energy_wms: Some(energy),
                latency_ms: latency,
                covered_latency_ms: latency,
                ..Default::default()
            };
            for failing_prefill in [true, false] {
                let summary = if failing_prefill {
                    TimingEvidenceSummary {
                        prefill: phase.clone(),
                        decode: phase.clone(),
                    }
                } else {
                    TimingEvidenceSummary {
                        prefill: TimingPhaseEvidence::default(),
                        decode: phase.clone(),
                    }
                };
                let source = TimingPowerSource {
                    timing: Arc::new(PowerTiming(summary)),
                    prefill_speedup_ratio: speedup,
                    decode_speedup_ratio: speedup,
                };
                // The final case overflows only when two phases are summed.
                if energy == f64::MAX && speedup == 1.0 && !failing_prefill {
                    continue;
                }
                let result = replay_timing_evidence(&[source]).and_then(|summary| {
                    let summary = summary.unwrap();
                    replay_power_diagnostics(Some(&summary), None)
                });
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn replay_power_is_withheld_below_aic_coverage_gate() {
        let source = TimingPowerSource {
            timing: Arc::new(PowerTiming(TimingEvidenceSummary {
                prefill: TimingPhaseEvidence {
                    energy_wms: Some(40_000.0),
                    latency_ms: 100.0,
                    covered_latency_ms: 80.0,
                    ..Default::default()
                },
                decode: TimingPhaseEvidence::default(),
            })),
            prefill_speedup_ratio: 1.0,
            decode_speedup_ratio: 1.0,
        };
        let evidence = replay_timing_evidence(&[source]).unwrap().unwrap();
        let power = replay_power_stats(&evidence).unwrap();
        assert_eq!(power.power_w, None);
        assert_eq!(power.coverage, 0.8);
    }

    #[test]
    fn replay_power_matches_shared_contract_fixtures() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/power-contract-v1.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut sources = Vec::new();
            for role in case["roles"].as_array().unwrap() {
                let timing: Arc<dyn TimingModel> = if role["energy_aware"].as_bool().unwrap() {
                    let ops = role["operations"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(index, op)| {
                            TimingOperationEvidence::new(
                                index.to_string(),
                                op["latency_ms"].as_f64().unwrap(),
                                op["energy_wms"].as_f64(),
                                TimingEvidenceSource::Silicon,
                            )
                            .unwrap()
                        })
                        .collect();
                    Arc::new(PowerTiming(TimingEvidenceSummary {
                        prefill: TimingPhaseEvidence::from_operations(ops),
                        decode: TimingPhaseEvidence::default(),
                    }))
                } else {
                    struct LatencyOnly;
                    impl TimingModel for LatencyOnly {
                        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> Result<f64> {
                            Ok(1.0)
                        }
                        fn predict_decode_ms(
                            &self,
                            _: usize,
                            _: usize,
                            _: usize,
                            _: usize,
                        ) -> Result<f64> {
                            Ok(1.0)
                        }
                    }
                    Arc::new(LatencyOnly)
                };
                let speedup = 1.0 / role["scale"].as_f64().unwrap_or(1.0);
                sources.push(TimingPowerSource {
                    timing,
                    prefill_speedup_ratio: speedup,
                    decode_speedup_ratio: speedup,
                });
            }
            let evidence = replay_timing_evidence(&sources).unwrap();
            let stats = evidence
                .as_ref()
                .map(replay_power_stats)
                .transpose()
                .unwrap();
            let diagnostics =
                serde_json::to_value(replay_power_diagnostics(evidence.as_ref(), None).unwrap())
                    .unwrap();
            assert_eq!(
                diagnostics["power_w"].is_null(),
                stats.and_then(|power| power.power_w).is_none()
            );
            assert_eq!(diagnostics["power_coverage"].is_null(), stats.is_none());
            let actual = [
                stats.and_then(|power| power.power_w),
                stats.map(|power| power.coverage),
            ];
            for (index, name) in ["power_w", "power_coverage"].iter().enumerate() {
                let expected = case["expected"][name].as_f64();
                match (actual[index], expected) {
                    (Some(actual), Some(expected)) => assert!(
                        (actual - expected).abs() < 1e-10,
                        "{} {name}: {actual} != {expected}",
                        case["name"]
                    ),
                    (None, None) => (),
                    _ => panic!(
                        "{} {name}: {:?} != {expected:?}",
                        case["name"], actual[index]
                    ),
                }
            }
        }
    }

    #[test]
    fn replay_power_is_published_at_the_exact_coverage_gate() {
        let source = TimingPowerSource {
            timing: Arc::new(PowerTiming(TimingEvidenceSummary {
                prefill: TimingPhaseEvidence {
                    energy_wms: Some(45_000.0),
                    latency_ms: 100.0,
                    covered_latency_ms: 90.0,
                    ..Default::default()
                },
                decode: TimingPhaseEvidence::default(),
            })),
            prefill_speedup_ratio: 1.0,
            decode_speedup_ratio: 1.0,
        };
        let evidence = replay_timing_evidence(&[source]).unwrap().unwrap();
        let power = replay_power_stats(&evidence).unwrap();
        assert_eq!(power.coverage, POWER_DATA_COVERAGE_THRESHOLD);
        assert_eq!(power.power_w, Some(450.0));
    }

    #[test]
    fn power_diagnostics_preserve_missingness_sources_and_reconciliation() {
        let summary = TimingEvidenceSummary {
            prefill: TimingPhaseEvidence::from_operations(vec![
                TimingOperationEvidence::new(
                    "gemm",
                    8.0,
                    Some(3_200.0),
                    TimingEvidenceSource::Silicon,
                )
                .unwrap(),
                TimingOperationEvidence::new(
                    "attention",
                    2.0,
                    None,
                    TimingEvidenceSource::Empirical,
                )
                .unwrap(),
            ]),
            decode: TimingPhaseEvidence::from_operations(vec![
                TimingOperationEvidence::new(
                    "moe",
                    4.0,
                    Some(1_600.0),
                    TimingEvidenceSource::from_provider("transferred"),
                )
                .unwrap(),
            ]),
        };

        let diagnostics = replay_power_diagnostics(Some(&summary), None).unwrap();
        let value = serde_json::to_value(diagnostics).unwrap();

        assert_eq!(value["publication_status"], "withheld");
        assert_eq!(value["power_coverage"], 12.0 / 14.0);
        assert_eq!(value["energy_wms"], 4_800.0);
        assert_eq!(value["latency_ms"], 14.0);
        assert_eq!(value["covered_latency_ms"], 12.0);
        assert_eq!(value.get("power_w"), Some(&serde_json::Value::Null));
        assert_eq!(value["phases"][0]["energy_wms"], 3_200.0);
        assert_eq!(value["phases"][1]["energy_wms"], 1_600.0);
        assert_eq!(value["phases"][1]["power_w"], 400.0);
        assert_eq!(value["phases"][1]["source_kind"], "transferred");
        let operations = value["phases"][0]["operations"].as_array().unwrap();
        assert_eq!(operations[0]["name"], "attention");
        assert!(operations[0].get("energy_wms").is_none());
        assert_eq!(operations[0]["status"], "missing");
        assert_eq!(operations[0]["source"], "empirical");
        assert_eq!(operations[1]["energy_contribution"], 1.0);

        let partial = operation_power_diagnostics(
            &TimingOperationEvidence {
                name: "partial".into(),
                energy_wms: Some(400.0),
                latency_ms: 2.0,
                covered_latency_ms: 1.0,
                source: TimingEvidenceSource::Mixed,
                details: None,
            },
            Some(400.0),
        );
        let partial = serde_json::to_value(partial).unwrap();
        assert_eq!(partial["status"], "partial");
        assert_eq!(partial["power_coverage"], 0.5);
        assert!(
            partial["uncovered_reason"]
                .as_str()
                .unwrap()
                .contains("some")
        );

        let zero_latency = operation_power_diagnostics(
            &TimingOperationEvidence {
                name: "zero-latency".into(),
                energy_wms: Some(400.0),
                latency_ms: 0.0,
                covered_latency_ms: 0.0,
                source: TimingEvidenceSource::Silicon,
                details: None,
            },
            Some(400.0),
        );
        let zero_latency = serde_json::to_value(zero_latency).unwrap();
        assert_eq!(zero_latency["status"], "available");
        assert_eq!(zero_latency["power_coverage"], 0.0);
        assert!(zero_latency.get("uncovered_reason").is_none());
    }

    #[test]
    fn performance_evidence_accumulates_sol_and_deduplicates_executed_fallbacks() {
        use crate::perfmodel::engine::diagnostics::{
            ExecutedFallback, OperationDetails, SolDiagnostics,
        };
        let mut op =
            TimingOperationEvidence::new("dispatch", 10.0, None, TimingEvidenceSource::Estimated)
                .unwrap();
        op.details = Some(OperationDetails {
            fpm_estimates: Vec::new(),
            sol: Some(SolDiagnostics {
                latency_ms: 2.0,
                math_ms: 0.0,
                memory_ms: 2.0,
            }),
            sol_unavailable_reason: None,
            fallbacks: vec![ExecutedFallback {
                inference_phase: "context".into(),
                comm_backend: "deepep_ll".into(),
                requested_ep_size: 32,
                requested_node_num: 8,
                measurement_ep_size: 4,
                measurement_node_num: 1,
            }],
        });
        let phase = TimingPhaseEvidence::try_from_operations(vec![op.clone()]).unwrap();
        let mut total = phase.clone();
        total.try_accumulate(phase).unwrap();
        // Two scheduled 10ms steps, divided by synthetic speedup 2. SOL is
        // the physical baseline: two 2ms steps, without synthetic speedup.
        let summary = TimingEvidenceSummary {
            prefill: scale_power_phase(total, 2.0).unwrap(),
            decode: Default::default(),
        };
        let report = replay_performance_diagnostics(Some(&summary));
        assert_eq!(report["phases"][0]["latency_ms"], 10.0);
        assert_eq!(report["phases"][0]["sol"]["latency_ms"], 4.0);
        assert_eq!(
            report["phases"][0]["operations"][0]["latency_to_sol_ratio"],
            2.5
        );
        assert_eq!(
            report["phases"][0]["operations"][0]["fallbacks"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        // A missing comparison in any scheduled step invalidates the total,
        // but does not hide its source or executed fallback records.
        op.details.as_mut().unwrap().sol = None;
        op.details.as_mut().unwrap().sol_unavailable_reason = Some("unsupported shape".into());
        let mut mixed = summary.prefill;
        mixed
            .try_accumulate(TimingPhaseEvidence::try_from_operations(vec![op]).unwrap())
            .unwrap();
        let report = replay_performance_diagnostics(Some(&TimingEvidenceSummary {
            prefill: mixed,
            decode: Default::default(),
        }));
        assert!(report["phases"][0]["sol"].is_null());
        assert_eq!(
            report["phases"][0]["operations"][0]["sol_unavailable_reason"],
            "unsupported shape"
        );
        assert_eq!(report["phases"][0]["operations"][0]["source"], "estimated");
        assert!(report["phases"][0]["operations"][0]["latency_to_sol_ratio"].is_null());
    }

    #[test]
    fn power_diagnostics_fail_closed_for_latency_only_providers() {
        let diagnostics = replay_power_diagnostics(
            None,
            Some("timing provider does not expose typed operation energy evidence"),
        )
        .unwrap();
        let value = serde_json::to_value(diagnostics).unwrap();

        assert_eq!(value["publication_status"], "unsupported");
        assert_eq!(value.get("power_w"), Some(&serde_json::Value::Null));
        assert_eq!(value.get("power_coverage"), Some(&serde_json::Value::Null));
        assert!(value.get("energy_wms").is_none());
        assert_eq!(
            value["unavailable_reason"],
            "timing provider does not expose typed operation energy evidence"
        );
        assert_eq!(value["phases"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn aic_timing_rejects_invalid_fpm_paths_before_entering_python() {
        for (path, model) in [("", "fpm"), ("/missing/fpm.parquet", "op_level")] {
            let mut config = aic_config();
            config.fpm_parquet_path = Some(path.into());
            config.forward_model = Some(model.into());
            let err = AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false)
                .err()
                .expect("invalid path");
            assert!(err.to_string().contains("fpm_parquet_path"), "{err}");
        }
    }

    #[test]
    fn aic_vision_timing_rejects_context_parallel_before_entering_python() {
        let mut config = aic_config();
        config.backend = "sglang".into();
        config.encoder_parallel = Some(crate::EncoderParallel::Tp);
        config.cp_size = Some(2);
        let err = AicTimingModel::build(&mut config, ForwardPassWorkerType::Prefill, true)
            .err()
            .expect("cp_size=2 vision rank");
        assert!(err.to_string().contains("cp_size=1"), "{err}");
    }

    #[test]
    fn aic_timing_config_accepts_fpm_forward_model() {
        let config = serde_json::from_value::<AicTimingConfig>(serde_json::json!({
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 1,
            "forward_model": "fpm",
            "fpm_profile": {"model": "test-model"},
            "fpm_interpolation": "direct",
            "fpm_parquet_path": "/artifacts/reviewed-fpm.parquet"
        }))
        .unwrap();
        assert_eq!(config.forward_model.as_deref(), Some("fpm"));
        assert_eq!(
            config.fpm_profile,
            Some(serde_json::from_value(serde_json::json!({"model": "test-model"})).unwrap())
        );
        assert_eq!(
            config.fpm_interpolation,
            Some(crate::FpmInterpolationMethod::Direct)
        );
        assert_eq!(
            config.fpm_parquet_path.as_deref(),
            Some("/artifacts/reviewed-fpm.parquet")
        );
    }

    #[test]
    fn timing_policy_reaches_canonical_request() {
        for replay in [false, true] {
            let config = serde_json::from_value::<AicTimingConfig>(serde_json::json!({
                "model": "test-model", "backend": "sglang", "system": "test-system", "tp": 1,
                "decoder_replay": replay, "database_mode": "SILICON",
                "forward_model": "fpm", "fpm_fmha_dtype": "fp8",
                "enable_shared_layer": false, "strict_provenance": true
            }))
            .unwrap();
            let request = config
                .estimator_request(ForwardPassWorkerType::Aggregated)
                .unwrap();
            assert_eq!(request.decoder_replay, replay);
            assert_eq!(request.fpm_fmha_quant_mode.as_deref(), Some("fp8"));
            assert_eq!(request.database_mode, crate::DatabaseMode::Silicon);
            assert_eq!(request.enable_shared_layer, Some(false));
            assert!(request.strict_provenance);
            let round_trip: ForwardPassPerfModelConfig =
                serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
            assert_eq!(round_trip, request);
        }
        let defaults = serde_json::from_value::<AicTimingConfig>(serde_json::json!({
            "model": "test-model", "backend": "sglang", "system": "test-system", "tp": 1
        }))
        .unwrap();
        assert!(!defaults.decoder_replay);
        assert_eq!(defaults.database_mode, crate::DatabaseMode::default());
        assert!(defaults.enable_shared_layer.is_none());
        assert!(!defaults.strict_provenance);
    }

    #[test]
    fn fpm_decode_timing_queries_exact_past_kv_total() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = python_timing_model(engine, true);

        let latency = timing
            .predict_decode_ms(35, 546_081, 15_602, 546_048)
            .unwrap();

        assert_eq!(latency, 546_046.0);
        assert_eq!(timing.evidence_summary(), None);
        assert_eq!(timing.fpm_evidence_summary(), None);
        // Capture is disabled: even a long sequence of distinct total-KV
        // queries cannot allocate a growing evidence history.
        for kv in 546_100..547_100 {
            timing.predict_decode_ms(35, kv, 15_602, 1_000_000).unwrap();
        }
        assert!(timing.fpm_evidence.lock().unwrap().is_empty());
    }

    #[test]
    fn prefill_totals_reach_fpm_exactly_and_op_level_as_means() {
        pyo3::prepare_freethreaded_python();
        let engine =
            || Python::with_gil(|py| Py::new(py, PrefillCoordinateProbe).unwrap().into_any());
        // Three requests computed 16 tokens after reading 8; the integer means
        // are (24 / 3, 8 / 3) = (8, 2).
        let fpm = python_timing_model(engine(), true);
        assert_eq!(
            fpm.predict_prefill_totals_ms(3, 16, 8).unwrap(),
            3_016_008.0
        );
        let op_level = python_timing_model(engine(), false);
        assert_eq!(
            op_level.predict_prefill_totals_ms(3, 16, 8).unwrap(),
            3_008_002.0
        );
    }

    #[test]
    fn native_fpm_prefill_queries_exact_totals_with_and_without_capture() {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        use crate::perfmodel::engine::runtime::tests::build_fpm_engine_with_rows;
        use crate::perfmodel::fpm::ForwardPassPerfOptions;
        pyo3::prepare_freethreaded_python();
        let row =
            |workload_kind, batch_size, total_prefill_tokens, total_kv_read_tokens, latency_ms| {
                RowSpec {
                    workload_kind,
                    batch_size,
                    total_prefill_tokens,
                    total_kv_read_tokens,
                    latency_ms,
                    ..RowSpec::default()
                }
            };
        let tmp = tempfile::tempdir().unwrap();
        let rows = [
            row("prefill", 1, 1, 0, 2.0),
            row("prefill", 3, 16, 8, 5.0),
            row("decode", 1, 0, 8, 3.0),
        ];
        let model = Arc::new(ForwardPassPerfModel::from_engine(
            Arc::new(build_fpm_engine_with_rows(tmp.path(), &rows).unwrap()),
            ForwardPassPerfOptions::default(),
        ));
        let engine = Python::with_gil(|py| Py::new(py, PrefillCoordinateProbe).unwrap().into_any());
        let mut timing = python_timing_model(engine, true);
        timing.fpm_model = Some(Arc::clone(&model));
        assert_eq!(timing.predict_prefill_totals_ms(3, 16, 8).unwrap(), 5.0);
        timing.diagnostic_model = Some(model);
        assert_eq!(timing.predict_prefill_totals_ms(3, 16, 8).unwrap(), 5.0);
        assert_eq!(
            timing.fpm_evidence_summary().unwrap().prefill.latency_ms,
            5.0
        );
        // The integer means of the same batch query (3, 18, 6), outside the
        // collected 16-token domain.
        let error = timing.predict_prefill_ms(3, 8, 2).unwrap_err();
        assert!(
            format!("{error:#}").contains("total_prefill_tokens=18"),
            "{error:#}"
        );
    }

    /// A native FPM model tuned so prefill of 3 requests (16 new tokens, 8 KV
    /// read) and decode of 1 request (8 past KV) observe 1.5x native latency.
    fn tuned_fpm_model(dir: &std::path::Path) -> Arc<ForwardPassPerfModel> {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        use crate::perfmodel::engine::runtime::tests::build_fpm_engine_with_rows;
        use crate::perfmodel::fpm::ForwardPassPerfOptions;
        let row =
            |workload_kind, batch_size, total_prefill_tokens, total_kv_read_tokens, latency_ms| {
                RowSpec {
                    workload_kind,
                    batch_size,
                    total_prefill_tokens,
                    total_kv_read_tokens,
                    latency_ms,
                    ..RowSpec::default()
                }
            };
        let rows = [
            row("prefill", 1, 1, 0, 2.0),
            row("prefill", 3, 16, 8, 5.0),
            row("decode", 1, 0, 8, 3.0),
        ];
        let mut model = ForwardPassPerfModel::from_engine(
            Arc::new(build_fpm_engine_with_rows(dir, &rows).unwrap()),
            ForwardPassPerfOptions::default(),
        );
        let observed = |scheduled_requests, native_ms: f64| {
            vec![crate::ForwardPassMetrics {
                wall_time: native_ms * 1.5 / 1000.0,
                scheduled_requests,
                ..Default::default()
            }]
        };
        let prefill = crate::ScheduledRequestMetrics {
            num_prefill_requests: 3,
            sum_prefill_tokens: 16,
            sum_prefill_kv_tokens: 8,
            ..Default::default()
        };
        let decode = crate::ScheduledRequestMetrics {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 8,
            ..Default::default()
        };
        let iterations: Vec<_> = (0..8)
            .flat_map(|_| {
                [
                    observed(prefill.clone(), 5.0),
                    observed(decode.clone(), 3.0),
                ]
            })
            .collect();
        model.tune_with_fpms(&iterations).unwrap();
        assert!(model.corrects_native_estimates());
        Arc::new(model)
    }

    fn corrected_estimate_ms(
        model: &ForwardPassPerfModel,
        scheduled_requests: crate::ScheduledRequestMetrics,
    ) -> f64 {
        model
            .estimate_forward_pass_time_ms(&[crate::ForwardPassMetrics {
                scheduled_requests,
                ..Default::default()
            }])
            .unwrap()
            .unwrap()
    }

    #[test]
    fn replay_applies_a_tuned_models_corrections_with_and_without_capture() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = tuned_fpm_model(tmp.path());
        let prefill = crate::ScheduledRequestMetrics {
            num_prefill_requests: 3,
            sum_prefill_tokens: 16,
            sum_prefill_kv_tokens: 8,
            ..Default::default()
        };
        let decode = crate::ScheduledRequestMetrics {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 8,
            ..Default::default()
        };
        let expected = (
            corrected_estimate_ms(&model, prefill),
            corrected_estimate_ms(&model, decode),
        );
        // The uncorrected engine prices these coordinates at 5 ms and 3 ms.
        assert!(expected.0 > 5.0 && expected.1 > 3.0, "{expected:?}");

        // Whole-model FPM: the same corrected timing with and without capture.
        let engine = Python::with_gil(|py| Py::new(py, PrefillCoordinateProbe).unwrap().into_any());
        let mut timing = python_timing_model(engine, true);
        timing.fpm_model = Some(Arc::clone(&model));
        timing.corrections = Some(Arc::clone(&model));
        let phases = |timing: &AicTimingModel| {
            (
                timing.predict_prefill_totals_ms(3, 16, 8).unwrap(),
                timing.predict_decode_ms(1, 9, 9, 1024).unwrap(),
            )
        };
        assert_eq!(phases(&timing), expected);
        timing.diagnostic_model = Some(Arc::clone(&model));
        assert_eq!(phases(&timing), expected);
        let evidence = timing.fpm_evidence_summary().unwrap();
        assert_eq!(
            (evidence.prefill.latency_ms, evidence.decode.latency_ms),
            expected
        );

        // Operation graphs: phase latency, evidence and energy scale together,
        // so power is unchanged.
        let op_engine = || {
            Python::with_gil(|py| {
                Py::new(py, PerOpEvidenceProbe::default())
                    .unwrap()
                    .into_any()
            })
        };
        let native = python_timing_model(op_engine(), false);
        let mut corrected = python_timing_model(op_engine(), false);
        corrected.corrections = Some(Arc::clone(&model));
        for timing in [&native, &corrected] {
            timing.predict_prefill_totals_ms(3, 16, 8).unwrap();
            timing.predict_decode_ms(1, 9, 9, 1024).unwrap();
        }
        let (native, corrected) = (
            native.evidence_summary().unwrap(),
            corrected.evidence_summary().unwrap(),
        );
        for (native, corrected, factor) in [
            (&native.prefill, &corrected.prefill, expected.0 / 5.0),
            (&native.decode, &corrected.decode, expected.1 / 3.0),
        ] {
            let power = |phase: &TimingPhaseEvidence| phase.energy_wms.unwrap() / phase.latency_ms;
            assert!((corrected.latency_ms / native.latency_ms - factor).abs() < 1e-12);
            assert!((power(corrected) / power(native) - 1.0).abs() < 1e-12);
            assert!(
                (corrected.covered_latency_ms / native.covered_latency_ms - factor).abs() < 1e-12
            );
        }
    }

    /// A native FPM model whose prefill row prices 8 new tokens at 10 ms and
    /// whose one-request decode rows price 4 and 12 past KV tokens at 2 and
    /// 4 ms. Decode at 8 past KV lies halfway between them (3 ms); the
    /// one-request pass baseline is the smallest-KV row (2 ms), so a decoder
    /// at 8 past KV adds 1 ms to a pass that computes that prefill.
    fn mixed_fpm_model(dir: &std::path::Path) -> ForwardPassPerfModel {
        use crate::perf_database::fpm_forward::tests::RowSpec;
        use crate::perfmodel::engine::runtime::tests::build_fpm_engine_with_rows;
        use crate::perfmodel::fpm::ForwardPassPerfOptions;
        let row =
            |workload_kind, batch_size, total_prefill_tokens, total_kv_read_tokens, latency_ms| {
                RowSpec {
                    workload_kind,
                    batch_size,
                    total_prefill_tokens,
                    total_kv_read_tokens,
                    latency_ms,
                    ..RowSpec::default()
                }
            };
        let rows = [
            row("prefill", 1, 8, 0, 10.0),
            row("decode", 1, 0, 4, 2.0),
            row("decode", 1, 0, 12, 4.0),
        ];
        ForwardPassPerfModel::from_engine(
            Arc::new(build_fpm_engine_with_rows(dir, &rows).unwrap()),
            ForwardPassPerfOptions::default(),
        )
    }

    /// The prefill of [`mixed_fpm_model`] alone, and with one decoder whose
    /// nine-token sequence has eight past KV tokens.
    fn mixed_fpm_iterations() -> (
        crate::ScheduledRequestMetrics,
        crate::ScheduledRequestMetrics,
    ) {
        let prefill = crate::ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 8,
            ..Default::default()
        };
        let mixed = crate::ScheduledRequestMetrics {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 8,
            ..prefill.clone()
        };
        (prefill, mixed)
    }

    const MIXED_FPM_PREFILL: PrefillTotals = PrefillTotals {
        batch_size: 1,
        new_tokens: 8,
        prefix_tokens: 0,
    };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn native_fpm_mixed_pass_is_the_one_pass_estimate() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(mixed_fpm_model(tmp.path()));
        let engine =
            || Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let mut timing = python_timing_model(engine(), true);
        timing.fpm_model = Some(Arc::clone(&model));
        let prefill = MIXED_FPM_PREFILL;
        let one_pass = corrected_estimate_ms(&model, mixed_fpm_iterations().1);
        let prefill_ms = timing.predict_prefill_totals_ms(1, 8, 0).unwrap();
        let decode_ms = timing.predict_decode_ms(1, 9, 9, 1024).unwrap();
        let pass = timing
            .predict_mixed_pass_ms(prefill, 1, 9, 9, 1024)
            .unwrap()
            .unwrap();
        // See `mixed_fpm_model`: 10 ms of prefill, a 3 ms decode pass, and a
        // decoder that adds 1 ms to the prefill's pass.
        assert!(close(prefill_ms, 10.0), "{prefill_ms}");
        assert!(close(decode_ms, 3.0), "{decode_ms}");
        assert!(close(one_pass, 11.0), "{one_pass}");
        assert_eq!(pass.prefill_ms, prefill_ms);
        assert!(close(pass.decode_ms, 1.0), "{pass:?}");
        assert_eq!(pass.prefill_ms + pass.decode_ms, one_pass);

        // Capture records the decode share apart from a decode pass at the
        // same totals.
        timing.diagnostic_model = Some(Arc::clone(&model));
        assert_eq!(
            timing
                .predict_mixed_pass_ms(prefill, 1, 9, 9, 1024)
                .unwrap(),
            Some(pass)
        );
        assert_eq!(timing.predict_decode_ms(1, 9, 9, 1024).unwrap(), decode_ms);
        let evidence = timing.fpm_evidence_summary().unwrap();
        assert_eq!(evidence.prefill.latency_ms, pass.prefill_ms);
        assert_eq!(evidence.decode.latency_ms, pass.decode_ms + decode_ms);

        // At the smallest collected KV the decode row is its own pass
        // baseline (2 ms each): the share is zero, yet both lookups ran and
        // stay in the captured evidence.
        assert_eq!(
            timing
                .predict_mixed_pass_ms(prefill, 1, 5, 5, 1024)
                .unwrap(),
            Some(MixedPassMs {
                prefill_ms,
                decode_ms: 0.0
            })
        );
        let uncorrected = 1.0_f64.to_bits();
        let history = timing.fpm_evidence.lock().unwrap();
        let zero_share = &history[&(FpmEvidencePhase::MixedDecode, 1, 0, 4, uncorrected)];
        assert_eq!(zero_share.count, 1);
        let rank = &zero_share.estimate.ranks[0];
        assert_eq!(
            (
                rank.decode_ms,
                rank.decode_baseline_ms,
                rank.marginal_decode_ms
            ),
            (Some(2.0), Some(2.0), Some(0.0))
        );
        // Both mixed passes computed the same prefill.
        assert_eq!(
            history[&(FpmEvidencePhase::Prefill, 1, 8, 0, uncorrected)].count,
            2
        );
        drop(history);

        // Operation graphs keep pricing a separate decode pass.
        let op_level = python_timing_model(engine(), false);
        assert_eq!(
            op_level
                .predict_mixed_pass_ms(prefill, 1, 9, 9, 1024)
                .unwrap(),
            None
        );
    }

    #[test]
    fn tuned_model_corrects_a_mixed_pass_as_one_mixed_iteration() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let mut model = mixed_fpm_model(tmp.path());
        let (prefill_only, mixed) = mixed_fpm_iterations();
        // The prefill alone runs at 1.5x its native 10 ms and the mixed
        // iteration at 1.25x its native one-pass 11 ms.
        let observed = |scheduled_requests, latency_ms: f64| {
            vec![crate::ForwardPassMetrics {
                wall_time: latency_ms / 1000.0,
                scheduled_requests,
                ..Default::default()
            }]
        };
        let iterations: Vec<_> = (0..8)
            .flat_map(|_| {
                [
                    observed(prefill_only.clone(), 15.0),
                    observed(mixed.clone(), 13.75),
                ]
            })
            .collect();
        model.tune_with_fpms(&iterations).unwrap();
        assert!(model.corrects_native_estimates());
        let model = Arc::new(model);
        let prefill_alone = corrected_estimate_ms(&model, prefill_only);
        let one_pass = corrected_estimate_ms(&model, mixed);
        assert!(close(prefill_alone, 15.0), "{prefill_alone}");
        assert!(close(one_pass, 13.75), "{one_pass}");

        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let mut timing = python_timing_model(engine, true);
        timing.fpm_model = Some(Arc::clone(&model));
        timing.corrections = Some(Arc::clone(&model));
        let check = |timing: &AicTimingModel| {
            // The pass costs the model's estimate of the mixed iteration: both
            // native parts (10 ms and 1 ms) take its 1.25x correction, not the
            // 1.5x of the prefill alone.
            let pass = timing
                .predict_mixed_pass_ms(MIXED_FPM_PREFILL, 1, 9, 9, 1024)
                .unwrap()
                .unwrap();
            assert!(close(pass.prefill_ms, 12.5), "{pass:?}");
            assert!(close(pass.decode_ms, 1.25), "{pass:?}");
            assert!(
                close(pass.prefill_ms + pass.decode_ms, one_pass),
                "{pass:?}"
            );
            // A pass that computes only that prefill keeps its own correction.
            assert_eq!(
                timing.predict_prefill_totals_ms(1, 8, 0).unwrap(),
                prefill_alone
            );
            pass
        };
        let pass = check(&timing);
        timing.diagnostic_model = Some(Arc::clone(&model));
        assert_eq!(check(&timing), pass);
        // Capture keeps the two corrections of the same prefill totals apart.
        let evidence = timing.fpm_evidence_summary().unwrap();
        assert!(
            close(evidence.prefill.latency_ms, pass.prefill_ms + prefill_alone),
            "{evidence:?}"
        );
        assert!(
            close(evidence.decode.latency_ms, pass.decode_ms),
            "{evidence:?}"
        );
    }

    #[test]
    fn fpm_decode_timing_caps_logical_past_kv_at_physical_capacity() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = python_timing_model(engine, true);

        let latency = timing
            .predict_decode_ms(35, 546_116, 15_603, 546_048)
            .unwrap();

        assert_eq!(latency, 546_048.0);
    }

    #[test]
    fn op_level_decode_timing_converts_inclusive_mean_to_past_kv() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = python_timing_model(engine, false);

        let latency = timing
            .predict_decode_ms(35, 546_081, 15_602, 546_048)
            .unwrap();

        assert_eq!(latency, 546_070.0);
        for inclusive_length in [1, 128, 129, 2049] {
            assert_eq!(
                timing
                    .predict_decode_ms(2, 2 * inclusive_length, inclusive_length, 8192)
                    .unwrap(),
                (2 * inclusive_length) as f64,
            );
        }
        assert!(timing.predict_decode_ms(1, 0, 0, 8192).is_err());
    }

    fn with_iteration_pricing(
        mut timing: AicTimingModel,
        model: &Arc<ForwardPassPerfModel>,
    ) -> AicTimingModel {
        // The selection `AicTimingModel::build` makes for this timing model.
        timing.iteration = IterationPricing::for_replay(
            model,
            &model.native_engine().unwrap(),
            timing.use_fpm_decode_totals,
            timing.decoder_replay,
        )
        .unwrap();
        assert!(
            timing.iteration.is_some(),
            "replay prices this model's iterations"
        );
        timing
    }

    fn prefill_work(dp_rank: u32, batch_size: usize, new_tokens: usize, prefix: usize) -> RankWork {
        RankWork {
            dp_rank,
            prefill: Some(crate::engine::PrefillWork {
                totals: PrefillTotals {
                    batch_size,
                    new_tokens,
                    prefix_tokens: prefix,
                },
                requests: Vec::new(),
            }),
            decode: None,
        }
    }

    /// `batch_size` requests whose contexts, each including the current
    /// token, total `active_kv_tokens`.
    fn decode_work(
        dp_rank: u32,
        batch_size: usize,
        active_kv_tokens: usize,
        kv_bound_tokens: usize,
    ) -> RankWork {
        RankWork {
            dp_rank,
            prefill: None,
            decode: Some(crate::engine::DecodeWork {
                batch_size,
                active_kv_tokens,
                mean_context_length: active_kv_tokens / batch_size,
                kv_bound_tokens,
                lengths: Vec::new(),
                verify_width: 1,
            }),
        }
    }

    fn mixed_work(prefill: RankWork, decode: RankWork) -> RankWork {
        RankWork {
            decode: decode.decode,
            ..prefill
        }
    }

    fn iteration_ms(timing: &AicTimingModel, ranks: &[RankWork]) -> f64 {
        timing
            .predict_iteration_ms(ranks)
            .unwrap()
            .expect("the canonical query prices this iteration")
            .total_ms
    }

    fn canonical_ms(model: &ForwardPassPerfModel, queries: &[RankQuery]) -> f64 {
        model
            .estimate_forward_pass_time_ms(
                &queries
                    .iter()
                    .map(AicTimingModel::rank_metrics)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .unwrap()
    }

    /// The phase-priced and iteration-priced adapters of one native FPM model.
    fn fpm_timing_pair(model: &Arc<ForwardPassPerfModel>) -> (AicTimingModel, AicTimingModel) {
        let timing = || {
            let engine =
                Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
            let mut timing = python_timing_model(engine, true);
            timing.fpm_model = Some(Arc::clone(model));
            timing.corrections = model.corrects_native_estimates().then(|| Arc::clone(model));
            timing
        };
        (timing(), with_iteration_pricing(timing(), model))
    }

    fn fpm_records(phase: &TimingPhaseEvidence) -> Vec<crate::FpmEstimateEvidence> {
        phase.operations.first().map_or_else(Vec::new, |operation| {
            operation.details.as_ref().unwrap().fpm_estimates.clone()
        })
    }

    #[test]
    fn fpm_iteration_matches_phase_pricing_for_one_rank_pure_passes() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let tuned_dir = tempfile::tempdir().unwrap();
        // Hand oracles: `mixed_fpm_model` prices 8 prefill tokens at 10 ms and
        // one decoder at 8 past KV at 3 ms. `tuned_fpm_model` prices 3
        // requests computing 16 tokens after 8 KV at 5 ms and one decoder at
        // 8 past KV at 3 ms, each under its own pure-phase correction. The
        // decoder's 9-token context includes its current token.
        for (model, (batch, new, prefix)) in [
            (Arc::new(mixed_fpm_model(tmp.path())), (1, 8, 0)),
            (tuned_fpm_model(tuned_dir.path()), (3, 16, 8)),
        ] {
            let (mut phase, mut iteration) = fpm_timing_pair(&model);
            for capture in [false, true] {
                if capture {
                    phase.diagnostic_model = Some(Arc::clone(&model));
                    iteration.diagnostic_model = Some(Arc::clone(&model));
                }
                let expected = phase.predict_prefill_totals_ms(batch, new, prefix).unwrap();
                let actual = iteration_ms(&iteration, &[prefill_work(0, batch, new, prefix)]);
                assert_eq!(actual.to_bits(), expected.to_bits(), "{actual} {expected}");
                let expected = phase.predict_decode_ms(1, 9, 9, 1024).unwrap();
                let actual = iteration_ms(&iteration, &[decode_work(0, 1, 9, 1024)]);
                assert_eq!(actual.to_bits(), expected.to_bits(), "{actual} {expected}");
            }
            // Captured once each: the same latency and estimate records.
            let (phase, iteration) = (
                phase.fpm_evidence_summary().unwrap(),
                iteration.fpm_evidence_summary().unwrap(),
            );
            for (phase, iteration) in [
                (&phase.prefill, &iteration.prefill),
                (&phase.decode, &iteration.decode),
            ] {
                assert_eq!(iteration.latency_ms.to_bits(), phase.latency_ms.to_bits());
                assert_eq!(fpm_records(iteration), fpm_records(phase));
                assert_eq!(fpm_records(iteration).len(), 1);
            }
        }
    }

    #[test]
    fn fpm_iteration_prices_a_mixed_rank_as_one_forward() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(mixed_fpm_model(tmp.path()));
        let (_, mut timing) = fpm_timing_pair(&model);
        timing.diagnostic_model = Some(Arc::clone(&model));
        let mixed = [mixed_work(
            prefill_work(0, 1, 8, 0),
            decode_work(0, 1, 9, 1024),
        )];
        // See `mixed_fpm_model`: 10 ms of prefill plus the 1 ms the decoder
        // adds to that forward.
        let iteration = iteration_ms(&timing, &mixed);
        assert!(close(iteration, 11.0), "{iteration}");
        assert_eq!(iteration, canonical_ms(&model, &[[1, 8, 0, 1, 8]]));
        assert_eq!(iteration_ms(&timing, &mixed), iteration);
        // The rank is one record, reported with decode, for each of the two
        // invocations.
        let evidence = timing.fpm_evidence_summary().unwrap();
        assert_eq!(evidence.prefill, TimingPhaseEvidence::default());
        assert!(close(evidence.decode.latency_ms, 2.0 * iteration));
        let records = fpm_records(&evidence.decode);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].count, 2);
        let rank = &records[0].estimate.ranks[0];
        assert_eq!(
            (rank.prefill_ms, rank.decode_ms, rank.marginal_decode_ms),
            (Some(10.0), Some(3.0), Some(1.0))
        );
    }

    #[test]
    fn fpm_iteration_records_each_working_ranks_estimate() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let mut model = mixed_fpm_model(tmp.path());
        // Corrections of 1.5 for the prefill alone and of the mixed iteration
        // for the decoder, the heaviest workload (see
        // `fpm_iteration_takes_the_slowest_rank_under_the_heaviest_ranks_correction`).
        let (prefill_only, mixed) = mixed_fpm_iterations();
        let observed = |scheduled_requests, latency_ms: f64| {
            vec![crate::ForwardPassMetrics {
                wall_time: latency_ms / 1000.0,
                scheduled_requests,
                ..Default::default()
            }]
        };
        let iterations: Vec<_> = (0..8)
            .flat_map(|_| {
                [
                    observed(prefill_only.clone(), 15.0),
                    observed(mixed.clone(), 13.75),
                ]
            })
            .collect();
        model.tune_with_fpms(&iterations).unwrap();
        let model = Arc::new(model);
        let (mut phase, mut timing) = fpm_timing_pair(&model);
        phase.diagnostic_model = Some(Arc::clone(&model));
        timing.diagnostic_model = Some(Arc::clone(&model));

        // DP=2: a prefill rank and a decode rank, twice, then the decode rank
        // with an idle sibling.
        let prefill = prefill_work(0, 1, 8, 0);
        let decode = decode_work(1, 1, 9, 1024);
        let grouped = iteration_ms(&timing, &[prefill.clone(), decode.clone()]);
        assert_eq!(iteration_ms(&timing, &[prefill, decode.clone()]), grouped);
        let alone = iteration_ms(&timing, &[RankWork::idle(0), decode]);
        let correction = model
            .native_correction_factor(&[AicTimingModel::rank_metrics(&[0, 0, 0, 1, 8])])
            .unwrap();
        // Not the prefill's own 1.5: the iteration takes one correction.
        assert!(!close(correction, 1.5), "{correction}");

        // Each working rank is its own record, at its own native latency
        // under the iteration's correction: the 10 ms prefill and the 3 ms
        // decode, as per-rank phase pricing records each rank's estimate.
        let evidence = timing.fpm_evidence_summary().unwrap();
        let prefill_records = fpm_records(&evidence.prefill);
        assert_eq!(prefill_records.len(), 1);
        assert_eq!(prefill_records[0].count, 2);
        let estimate = &prefill_records[0].estimate;
        assert_eq!(
            (estimate.native_latency_ms, estimate.correction_factor),
            (Some(10.0), Some(correction))
        );
        assert!(close(estimate.latency_ms.unwrap(), 10.0 * correction));
        assert_eq!(estimate.max_rank, Some(0));
        assert_eq!(estimate.ranks.len(), 1);
        assert_eq!(estimate.ranks[0].rank, 0);
        assert!(close(grouped, 10.0 * correction), "{grouped}");
        // The decode rank's record matches its one-rank phase estimate, whose
        // correction the grouped and lone iterations share: one record of
        // three invocations, the sum of its latencies.
        let decode_records = fpm_records(&evidence.decode);
        assert_eq!(decode_records.len(), 1);
        assert_eq!(decode_records[0].count, 3);
        assert_eq!(alone, phase.predict_decode_ms(1, 9, 9, 1024).unwrap());
        let phase_records = fpm_records(&phase.fpm_evidence_summary().unwrap().decode);
        assert_eq!(decode_records[0].estimate, phase_records[0].estimate);
        assert!(close(evidence.decode.latency_ms, 3.0 * alone));
        assert!(close(
            evidence.prefill.latency_ms,
            2.0 * estimate.latency_ms.unwrap()
        ));
    }

    #[test]
    fn fpm_iteration_takes_the_slowest_rank_under_the_heaviest_ranks_correction() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let mut model = mixed_fpm_model(tmp.path());
        // The prefill alone runs at 1.5x its native 10 ms (see
        // `tuned_model_corrects_a_mixed_pass_as_one_mixed_iteration`).
        let (prefill_only, mixed) = mixed_fpm_iterations();
        let observed = |scheduled_requests, latency_ms: f64| {
            vec![crate::ForwardPassMetrics {
                wall_time: latency_ms / 1000.0,
                scheduled_requests,
                ..Default::default()
            }]
        };
        let iterations: Vec<_> = (0..8)
            .flat_map(|_| {
                [
                    observed(prefill_only.clone(), 15.0),
                    observed(mixed.clone(), 13.75),
                ]
            })
            .collect();
        model.tune_with_fpms(&iterations).unwrap();
        let model = Arc::new(model);
        let (phase, timing) = fpm_timing_pair(&model);

        // A prefill rank and an idle rank: the prefill's own correction.
        let alone = iteration_ms(&timing, &[prefill_work(0, 1, 8, 0), RankWork::idle(1)]);
        assert!(close(alone, 15.0), "{alone}");
        assert_eq!(alone, phase.predict_prefill_totals_ms(1, 8, 0).unwrap());

        // Imbalanced ranks: the iteration is the slowest rank's native 10 ms
        // prefill under the correction of the heaviest rank's workload, the
        // decoder's (1 request + 8 KV outweighs 8 prefill tokens). Phase
        // pricing instead took the maximum of each rank's own corrected time.
        let ranks = [prefill_work(0, 1, 8, 0), decode_work(1, 1, 9, 1024)];
        let grouped = iteration_ms(&timing, &ranks);
        assert_eq!(
            grouped,
            canonical_ms(&model, &[[1, 8, 0, 0, 0], [0, 0, 0, 1, 8]])
        );
        let decode_factor = model
            .native_correction_factor(&[AicTimingModel::rank_metrics(&[0, 0, 0, 1, 8])])
            .unwrap();
        assert!(
            close(grouped, 10.0 * decode_factor),
            "{grouped} {decode_factor}"
        );
        let per_rank = phase
            .predict_prefill_totals_ms(1, 8, 0)
            .unwrap()
            .max(phase.predict_decode_ms(1, 9, 9, 1024).unwrap());
        assert!(close(per_rank, 15.0), "{per_rank}");
        assert!(!close(grouped, per_rank), "{grouped} {per_rank}");
    }

    #[test]
    fn iteration_cache_hits_only_when_every_rank_matches() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(mixed_fpm_model(tmp.path()));
        let (_, mut timing) = fpm_timing_pair(&model);
        timing.diagnostic_model = Some(Arc::clone(&model));
        let cached = |timing: &AicTimingModel| {
            timing
                .iteration
                .as_ref()
                .unwrap()
                .cache
                .as_ref()
                .unwrap()
                .len()
        };
        let prefill = || prefill_work(0, 1, 8, 0);
        let decode = |dp_rank| decode_work(dp_rank, 1, 9, 1024);
        let first = iteration_ms(&timing, &[prefill(), decode(1)]);
        assert_eq!(cached(&timing), 1);
        assert_eq!(iteration_ms(&timing, &[prefill(), decode(1)]), first);
        assert_eq!(cached(&timing), 1, "an identical rank list hits");
        // Every rank, in order, is part of the key.
        let mut reversed = [decode(0), prefill()];
        reversed[1].dp_rank = 1;
        iteration_ms(&timing, &reversed);
        assert_eq!(cached(&timing), 2);
        iteration_ms(&timing, &[prefill(), RankWork::idle(1)]);
        assert_eq!(cached(&timing), 3);
        // A hit still records its invocation, once for each working rank.
        let records = timing.fpm_iteration_evidence.lock().unwrap();
        let counts = records
            .iter()
            .map(|((kind, query, _), record)| (*kind, *query, record.count))
            .collect::<Vec<_>>();
        assert_eq!(
            counts,
            [
                (IterationKind::Prefill, [1, 8, 0, 0, 0], 4),
                (IterationKind::Decode, [0, 0, 0, 1, 8], 3),
            ]
        );
    }

    #[test]
    fn iteration_pricing_declines_what_the_canonical_query_cannot_represent() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(mixed_fpm_model(tmp.path()));
        let (phase, mut timing) = fpm_timing_pair(&model);
        let pure = [prefill_work(0, 1, 8, 0)];
        let mixed = [mixed_work(
            prefill_work(0, 1, 8, 0),
            decode_work(0, 1, 9, 1024),
        )];
        // Providers built without iteration pricing keep per-rank phases.
        assert_eq!(phase.predict_iteration_ms(&pure).unwrap(), None);
        // An all-idle group computes nothing.
        assert_eq!(
            timing
                .predict_iteration_ms(&[RankWork::idle(0), RankWork::idle(1)])
                .unwrap(),
            Some(IterationMs { total_ms: 0.0 })
        );
        // Totals beyond the query's u32 range keep phase pricing.
        assert_eq!(
            timing
                .predict_iteration_ms(&[prefill_work(0, 1, u32::MAX as usize + 1, 0)])
                .unwrap(),
            None
        );
        let pricing = timing.iteration.as_mut().unwrap();
        // A measured SD-round profile prices exactly one rank, mixed or pure
        // (see `sd_round_iteration_prices_a_mixed_rank_as_one_forward`).
        pricing.query = IterationQuery::Native(IterationDispatch::WholeModel { sd_round: true });
        assert!(timing.predict_iteration_ms(&pure).unwrap().is_some());
        assert!(timing.predict_iteration_ms(&mixed).unwrap().is_some());
        let two_ranks = [prefill_work(0, 1, 8, 0), RankWork::idle(1)];
        assert_eq!(timing.predict_iteration_ms(&two_ranks).unwrap(), None);
        // Measured scopes without a mix-step composition decline mixed ranks,
        // and a scheduler at another speculative depth keeps phase pricing.
        let pricing = timing.iteration.as_mut().unwrap();
        pricing.query = IterationQuery::Native(IterationDispatch::Operations {
            verify_width: 1,
            mixed: false,
        });
        assert_eq!(timing.predict_iteration_ms(&mixed).unwrap(), None);
        let pricing = timing.iteration.as_mut().unwrap();
        pricing.query = IterationQuery::Native(IterationDispatch::Operations {
            verify_width: 2,
            mixed: true,
        });
        assert_eq!(
            timing
                .predict_iteration_ms(&[decode_work(0, 1, 9, 1024)])
                .unwrap(),
            None
        );
    }

    #[test]
    fn replay_prices_whole_model_fpm_iterations_except_under_decoder_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(mixed_fpm_model(tmp.path()));
        let native = model.native_engine().unwrap();
        let dispatch = |use_fpm_decode_totals, decoder_replay| {
            IterationPricing::for_replay(&model, &native, use_fpm_decode_totals, decoder_replay)
                .unwrap()
                .map(|pricing| pricing.query)
        };
        // Whole-model FPM timing reads decode totals from this engine.
        assert_eq!(
            dispatch(true, false),
            Some(IterationQuery::Native(IterationDispatch::WholeModel {
                sd_round: false
            }))
        );
        assert_eq!(dispatch(true, true), None);
        assert_eq!(dispatch(false, false), None);
        // The whole-model engine composes no operation-graph mix step.
        assert!(native.mix_step_passes().is_none());
    }

    #[test]
    fn rank_queries_follow_each_estimators_decode_convention() {
        let mut decode = decode_work(0, 2, 200, 150);
        decode.decode.as_mut().unwrap().verify_width = 3;
        // Whole-model FPM: 2 requests reading their 198 past tokens, clamped
        // to the 150-token bound, whatever the verification width.
        assert_eq!(
            AicTimingModel::rank_query(IterationDispatch::WholeModel { sd_round: false }, &decode)
                .unwrap(),
            Some([0, 0, 0, 2, 150])
        );
        // Operation graphs: 2 requests widened to 6 verified tokens, each at
        // the 100-token mean context including the current token.
        assert_eq!(
            AicTimingModel::rank_query(
                IterationDispatch::Operations {
                    verify_width: 3,
                    mixed: true
                },
                &mixed_work(prefill_work(0, 3, 16, 8), decode)
            )
            .unwrap(),
            Some([3, 16, 8, 6, 600])
        );
        let mut empty_context = decode_work(0, 1, 0, 8);
        empty_context.decode.as_mut().unwrap().mean_context_length = 0;
        for dispatch in [
            IterationDispatch::WholeModel { sd_round: false },
            IterationDispatch::Operations {
                verify_width: 1,
                mixed: true,
            },
        ] {
            assert!(AicTimingModel::rank_query(dispatch, &empty_context).is_err());
        }
    }

    /// A packaged Qwen3-0.6B operation-level model, or `None` when the
    /// embedded interpreter cannot import the Python SDK. Without `power`, the
    /// H200 vLLM model, whose tables carry no power; with it, the B200
    /// TensorRT-LLM 1.3.0rc20 model, whose GEMM and attention tables do.
    fn packaged_op_level_timing(test: &str, power: bool) -> Option<AicTimingModel> {
        let required = std::env::var_os("AIC_REQUIRE_EMBEDDED_ROUND_TRIP").is_some();
        if let Err(error) =
            Python::with_gil(|py| py.import("aisimulate_core.sdk.engine").map(|_| ()))
        {
            assert!(
                !required,
                "{test}: AIC_REQUIRE_EMBEDDED_ROUND_TRIP is set but `aisimulate_core.sdk.engine` \
                 is not importable: {error}"
            );
            eprintln!("{test}: SKIP — `aisimulate_core.sdk.engine` is not importable: {error}");
            return None;
        }
        let mut config = aic_config();
        config.model = "Qwen/Qwen3-0.6B".into();
        if power {
            config.system = "b200_sxm".into();
            config.backend = "trtllm".into();
            config.backend_version = Some("1.3.0rc20".into());
        } else {
            config.system = "h200_sxm".into();
        }
        config.estimation_mode = Some(EstimationMode::OpLevel);
        config.kv_block_size = Some(16);
        config.systems_paths = vec![
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems"),
        ];
        let mut timing =
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated, false).unwrap();
        timing.diagnostic_model = None;
        assert!(matches!(
            timing.iteration.as_ref().unwrap().query,
            IterationQuery::Native(IterationDispatch::Operations {
                verify_width: 1,
                mixed: true
            })
        ));
        Some(timing)
    }

    /// Latency and operation evidence of one pricing on a fresh epoch.
    fn priced_with_evidence(
        timing: &AicTimingModel,
        price: impl FnOnce(&AicTimingModel) -> f64,
    ) -> (f64, TimingEvidenceSummary) {
        timing.reset_evidence().unwrap();
        let latency_ms = price(timing);
        (latency_ms, timing.evidence_summary().unwrap())
    }

    #[test]
    fn op_level_iteration_on_a_packaged_model_keeps_pure_one_rank_phase_timing() {
        pyo3::prepare_freethreaded_python();
        for power in [false, true] {
            let Some(timing) = packaged_op_level_timing(
                "op_level_iteration_on_a_packaged_model_keeps_pure_one_rank_phase_timing",
                power,
            ) else {
                return;
            };
            assert_pure_one_rank_iterations_keep_phase_timing(timing, power);
        }
    }

    fn assert_pure_one_rank_iterations_keep_phase_timing(mut timing: AicTimingModel, power: bool) {
        let model = Arc::clone(&timing.iteration.as_ref().unwrap().model);
        for capture in [false, true] {
            // Replay fixes capture at construction; each mode prices and
            // caches its own evidence.
            timing.diagnostic_model = capture.then(|| Arc::clone(&model));
            timing.phase_cache.clear();
            let pricing = timing.iteration.as_ref().unwrap();
            pricing.cache.as_ref().unwrap().clear();
            let timing = &timing;
            // Pure passes whose per-request means divide exactly: the
            // canonical query prices the same static phase, and the rank's
            // operations are the same evidence, with the same details while
            // diagnostics are captured.
            for (batch, new, prefix) in [(1, 1000, 0), (4, 4096, 1024), (16, 2048, 0)] {
                let iteration = priced_with_evidence(timing, |timing| {
                    iteration_ms(timing, &[prefill_work(0, batch, new, prefix)])
                });
                let phase = priced_with_evidence(timing, |timing| {
                    timing
                        .predict_prefill_totals_ms(batch, new, prefix)
                        .unwrap()
                });
                assert_eq!(
                    iteration.0.to_bits(),
                    phase.0.to_bits(),
                    "{iteration:?} {phase:?}"
                );
                assert_eq!(iteration.1, phase.1);
                assert_eq!(iteration.1.prefill.energy_wms.is_some(), power);
                let details = |operation: &TimingOperationEvidence| operation.details.is_some();
                assert!(
                    iteration
                        .1
                        .prefill
                        .operations
                        .iter()
                        .all(|op| details(op) == capture)
                );
            }
            for (batch, context) in [(1, 129), (32, 1025), (7, 4097)] {
                let iteration = priced_with_evidence(timing, |timing| {
                    iteration_ms(timing, &[decode_work(0, batch, batch * context, 1 << 20)])
                });
                let phase = priced_with_evidence(timing, |timing| {
                    timing
                        .predict_decode_ms(batch, batch * context, context, 1 << 20)
                        .unwrap()
                });
                assert_eq!(
                    iteration.0.to_bits(),
                    phase.0.to_bits(),
                    "{iteration:?} {phase:?}"
                );
                assert_eq!(iteration.1, phase.1);
                assert_eq!(iteration.1.decode.energy_wms.is_some(), power);
            }
        }
    }

    /// Operation evidence of `operations` as one phase record would carry it,
    /// without details.
    fn operations_evidence(
        operations: &[crate::perfmodel::engine::diagnostics::StaticOperationDiagnostics],
    ) -> TimingPhaseEvidence {
        TimingPhaseEvidence::from_operations(
            operations
                .iter()
                .map(|operation| {
                    TimingOperationEvidence::new(
                        operation.name.clone(),
                        operation.latency_ms,
                        Some(operation.energy_wms),
                        TimingEvidenceSource::from_provider(operation.source.clone()),
                    )
                    .unwrap()
                })
                .collect(),
        )
    }

    #[test]
    fn op_level_iteration_on_a_packaged_model_uses_the_canonical_composition() {
        pyo3::prepare_freethreaded_python();
        let Some(timing) = packaged_op_level_timing(
            "op_level_iteration_on_a_packaged_model_uses_the_canonical_composition",
            true,
        ) else {
            return;
        };
        let model = Arc::clone(&timing.iteration.as_ref().unwrap().model);

        // Prefill means: the canonical query divides each total by the batch
        // (2049 / 2 = 1024 new tokens after 1023 / 2 = 511), where phase
        // pricing kept (2049 + 1023) / 2 - 511 = 1025.
        let iteration = iteration_ms(&timing, &[prefill_work(0, 2, 2049, 1023)]);
        assert_eq!(iteration, canonical_ms(&model, &[[2, 2049, 1023, 0, 0]]));
        assert_eq!(iteration, canonical_ms(&model, &[[2, 2048, 1022, 0, 0]]));
        assert_ne!(
            iteration,
            timing.predict_prefill_totals_ms(2, 2049, 1023).unwrap()
        );

        // A mixed vLLM pass is the model's one mix step, not a prefill pass
        // followed by a separate decode pass.
        let query = [1, 512, 0, 8, 8 * 1025];
        let decode = decode_work(0, 8, 8 * 1025, 1 << 20);
        let mixed = [mixed_work(prefill_work(0, 1, 512, 0), decode.clone())];
        let (iteration, evidence) =
            priced_with_evidence(&timing, |timing| iteration_ms(timing, &mixed));
        assert_eq!(iteration, canonical_ms(&model, &[query]));
        let separate = timing.predict_prefill_totals_ms(1, 512, 0).unwrap()
            + timing
                .predict_decode_ms(8, 8 * 1025, 1025, 1 << 20)
                .unwrap();
        assert_ne!(iteration, separate);
        let rank = model
            .estimate_forward_pass_detailed(&[AicTimingModel::rank_metrics(&query)])
            .unwrap()
            .ranks
            .remove(0);
        let mix_step = operations_evidence(&rank.operations);
        assert_mixed_rank_attributed_by_pass(&evidence, &mix_step, 512.0 / (512.0 + 8.0));
        assert!(
            (evidence.prefill.latency_ms + evidence.decode.latency_ms - iteration).abs()
                <= 1e-12 * iteration
        );
    }

    /// `evidence` is one mixed rank's `mix_step` operations attributed by
    /// pass: context attention is prefill, generation attention decode, and
    /// each other operation, all from the shared non-attention pass, splits
    /// `token_share` prefill. Each part keeps its operation's power, and the
    /// rank's latency and energy are unchanged.
    fn assert_mixed_rank_attributed_by_pass(
        evidence: &TimingEvidenceSummary,
        mix_step: &TimingPhaseEvidence,
        token_share: f64,
    ) {
        let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        let find = |phase: &TimingPhaseEvidence, name: &str| {
            phase
                .operations
                .iter()
                .find(|operation| operation.name == name)
                .map(|operation| (operation.latency_ms, operation.energy_wms.unwrap_or(0.0)))
        };
        let names = mix_step
            .operations
            .iter()
            .map(|operation| operation.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"context_attention") && names.contains(&"generation_attention"));
        assert!(names.len() > 2, "{names:?}");
        for operation in &mix_step.operations {
            let fraction = match operation.name.as_str() {
                "context_attention" => 1.0,
                "generation_attention" => 0.0,
                _ => token_share,
            };
            let energy = operation.energy_wms.unwrap_or(0.0);
            for (phase, part) in [
                (&evidence.prefill, fraction),
                (&evidence.decode, 1.0 - fraction),
            ] {
                match find(phase, &operation.name) {
                    Some((latency_ms, energy_wms)) => {
                        assert!(near(latency_ms, part * operation.latency_ms));
                        assert!(near(energy_wms, part * energy));
                    }
                    None => assert_eq!(part, 0.0, "{}", operation.name),
                }
            }
        }
        let energy = |phase: &TimingPhaseEvidence| phase.energy_wms.unwrap();
        assert!(near(
            evidence.prefill.latency_ms + evidence.decode.latency_ms,
            mix_step.latency_ms
        ));
        assert!(near(
            energy(&evidence.prefill) + energy(&evidence.decode),
            energy(mix_step)
        ));
    }

    #[test]
    fn a_mixed_rank_splits_its_operations_by_pass_or_wholly_by_token_share() {
        use crate::perfmodel::engine::diagnostics::{OperationDetails, StaticOperationDiagnostics};
        let operation = |name: &str| StaticOperationDiagnostics {
            name: name.into(),
            latency_ms: 1.0,
            energy_wms: 1.0,
            source: "silicon".into(),
            details: OperationDetails {
                sol: None,
                sol_unavailable_reason: None,
                fallbacks: Vec::new(),
                fpm_estimates: Vec::new(),
            },
        };
        let operations = ["qkv_gemm", "context_attention", "generation_attention"].map(operation);
        // 30 prefill tokens and 10 packed decode requests: 3/4 prefill.
        let query = [2, 30, 0, 10, 1000];
        let mut passes = std::collections::HashMap::from([
            ("qkv_gemm".to_owned(), MixedPass::SharedNonAttention),
            ("context_attention".to_owned(), MixedPass::ContextAttention),
            (
                "generation_attention".to_owned(),
                MixedPass::DecodeAttention,
            ),
        ]);
        let fractions = |kind, passes: Option<&_>| {
            AicTimingModel::prefill_fractions(kind, &query, &operations, passes)
        };
        assert_eq!(
            fractions(IterationKind::Mixed, Some(&passes)),
            [0.75, 1.0, 0.0]
        );
        // Pure ranks are wholly their phase.
        assert_eq!(fractions(IterationKind::Prefill, Some(&passes)), [1.0; 3]);
        assert_eq!(fractions(IterationKind::Decode, None), [0.0; 3]);
        // Without every operation's pass, the whole rank splits by its token
        // share.
        assert_eq!(fractions(IterationKind::Mixed, None), [0.75; 3]);
        passes.remove("generation_attention");
        assert_eq!(fractions(IterationKind::Mixed, Some(&passes)), [0.75; 3]);
    }

    #[test]
    fn tuned_op_level_decode_takes_its_correction_at_the_packed_coordinates() {
        pyo3::prepare_freethreaded_python();
        let Some(mut timing) = packaged_op_level_timing(
            "tuned_op_level_decode_takes_its_correction_at_the_packed_coordinates",
            false,
        ) else {
            return;
        };
        // 100 requests at a 5000-token context including the current token:
        // the canonical query reads 100 requests at 100 * 5000 tokens, where
        // per-rank phase pricing looked its correction up at their 100 * 4999
        // past KV tokens. The default correction grid divides decode KV into
        // four regions of 2,000,000 / 4 = 500,000 tokens, so the two lookups
        // fall into different regions.
        let (batch, context) = (100_u32, 5000_u32);
        let packed = [0, 0, 0, batch, batch * context];
        let past = [0, 0, 0, batch, batch * (context - 1)];
        let work = [decode_work(
            0,
            batch as usize,
            (batch * context) as usize,
            1 << 20,
        )];
        let native = iteration_ms(&timing, &work);
        let phase_native = timing
            .predict_decode_ms(
                batch as usize,
                (batch * context) as usize,
                context as usize,
                1 << 20,
            )
            .unwrap();
        assert_eq!(native, phase_native);
        // Observe 1.25x at the packed coordinates only.
        let pricing = timing.iteration.as_mut().unwrap();
        let tuned = Arc::get_mut(&mut pricing.model).expect("the fixture's only model handle");
        let iterations: Vec<_> = (0..8)
            .map(|_| {
                vec![crate::ForwardPassMetrics {
                    wall_time: native * 1.25 / 1000.0,
                    ..AicTimingModel::rank_metrics(&packed)
                }]
            })
            .collect();
        tuned.tune_with_fpms(&iterations).unwrap();
        let model = Arc::clone(&pricing.model);
        pricing.cache.as_ref().unwrap().clear();
        timing.corrections = Some(Arc::clone(&model));
        let factor = |query: [u32; 5]| {
            model
                .native_correction_factor(&[AicTimingModel::rank_metrics(&query)])
                .unwrap()
        };
        assert!(close(factor(packed), 1.25), "{}", factor(packed));
        assert!(!close(factor(past), factor(packed)), "{}", factor(past));
        // The iteration takes the packed coordinates' correction, the phase
        // path the past-KV coordinates'.
        let iteration = iteration_ms(&timing, &work);
        assert_eq!(iteration, canonical_ms(&model, &[packed]));
        assert!(close(iteration, 1.25 * native), "{iteration} {native}");
        let phase = timing
            .predict_decode_ms(
                batch as usize,
                (batch * context) as usize,
                context as usize,
                1 << 20,
            )
            .unwrap();
        assert!(close(phase, factor(past) * native), "{phase} {native}");
    }

    #[test]
    fn op_level_iteration_records_every_ranks_operation_evidence() {
        pyo3::prepare_freethreaded_python();
        let Some(timing) = packaged_op_level_timing(
            "op_level_iteration_records_every_ranks_operation_evidence",
            true,
        ) else {
            return;
        };
        let model = Arc::clone(&timing.iteration.as_ref().unwrap().model);

        // Imbalanced decode ranks: one iteration at the slower rank's time,
        // and the energy of both ranks' work, as when each rank recorded its
        // own decode pass.
        let light = decode_work(0, 4, 4 * 1025, 1 << 20);
        let heavy = decode_work(1, 32, 32 * 4097, 1 << 20);
        let (grouped, evidence) = priced_with_evidence(&timing, |timing| {
            iteration_ms(timing, &[light.clone(), heavy.clone()])
        });
        let single = |work: &RankWork| {
            let mut work = work.clone();
            work.dp_rank = 0;
            priced_with_evidence(&timing, |timing| iteration_ms(timing, &[work]))
        };
        let ((light_ms, light_evidence), (heavy_ms, heavy_evidence)) =
            (single(&light), single(&heavy));
        assert_eq!(grouped, light_ms.max(heavy_ms));
        assert_eq!(
            grouped,
            canonical_ms(&model, &[[0, 0, 0, 4, 4 * 1025], [0, 0, 0, 32, 32 * 4097]])
        );
        let (_, per_rank) = priced_with_evidence(&timing, |timing| {
            timing
                .predict_decode_ms(4, 4 * 1025, 1025, 1 << 20)
                .unwrap()
                + timing
                    .predict_decode_ms(32, 32 * 4097, 4097, 1 << 20)
                    .unwrap()
        });
        assert_eq!(evidence, per_rank);
        let energy = |phase: &TimingPhaseEvidence| phase.energy_wms.unwrap();
        let sum = energy(&light_evidence.decode) + energy(&heavy_evidence.decode);
        assert!((energy(&evidence.decode) - sum).abs() <= 1e-12 * sum);
        assert!(energy(&evidence.decode) > energy(&heavy_evidence.decode));
        assert!(
            (evidence.decode.latency_ms
                - (light_evidence.decode.latency_ms + heavy_evidence.decode.latency_ms))
                .abs()
                <= 1e-12 * evidence.decode.latency_ms
        );

        // A prefill rank and a decode rank: each rank's work is recorded with
        // its own phase, so each takes that phase's speedup.
        let prefill = prefill_work(0, 4, 4096, 1024);
        let decode = decode_work(1, 32, 32 * 1025, 1 << 20);
        let (_, evidence) = priced_with_evidence(&timing, |timing| {
            iteration_ms(timing, &[prefill.clone(), decode.clone()])
        });
        let (_, per_rank) = priced_with_evidence(&timing, |timing| {
            timing.predict_prefill_totals_ms(4, 4096, 1024).unwrap()
                + timing
                    .predict_decode_ms(32, 32 * 1025, 1025, 1 << 20)
                    .unwrap()
        });
        assert_eq!(evidence, per_rank);
        assert!(evidence.prefill.energy_wms.is_some() && evidence.decode.energy_wms.is_some());

        // An idle rank records nothing.
        let (_, evidence) = priced_with_evidence(&timing, |timing| {
            iteration_ms(timing, &[RankWork::idle(0), decode.clone()])
        });
        let (_, alone) = priced_with_evidence(&timing, |timing| {
            iteration_ms(
                timing,
                &[RankWork {
                    dp_rank: 0,
                    ..decode.clone()
                }],
            )
        });
        assert_eq!(evidence, alone);

        // Tune the model so the prefill rank alone runs at 1.5x and the decode
        // rank alone at 1.25x its native time.
        let untuned = per_rank;
        let queries = [[4, 4096, 1024, 0, 0], [0, 0, 0, 32, 32 * 1025]];
        let natives = queries.map(|query| canonical_ms(&model, &[query]));
        drop(model);
        let mut timing = timing;
        let pricing = timing.iteration.as_mut().unwrap();
        let tuned = Arc::get_mut(&mut pricing.model).expect("the fixture's only model handle");
        let iterations: Vec<_> = (0..8)
            .flat_map(|_| {
                [(0, 1.5), (1, 1.25)].map(|(index, factor)| {
                    vec![crate::ForwardPassMetrics {
                        wall_time: natives[index] * factor / 1000.0,
                        ..AicTimingModel::rank_metrics(&queries[index])
                    }]
                })
            })
            .collect();
        tuned.tune_with_fpms(&iterations).unwrap();
        let model = Arc::clone(&pricing.model);
        pricing.cache.as_ref().unwrap().clear();
        assert!(model.corrects_native_estimates());
        timing.corrections = Some(Arc::clone(&model));

        // One rank: the phase path's corrected evidence.
        let (corrected_prefill, evidence) = priced_with_evidence(&timing, |timing| {
            iteration_ms(
                timing,
                &[RankWork {
                    dp_rank: 0,
                    ..prefill.clone()
                }],
            )
        });
        let (phase_prefill, phase) = priced_with_evidence(&timing, |timing| {
            timing.predict_prefill_totals_ms(4, 4096, 1024).unwrap()
        });
        assert_eq!(corrected_prefill, phase_prefill);
        assert!(
            close(corrected_prefill, 1.5 * natives[0]),
            "{corrected_prefill}"
        );
        assert_eq!(evidence, phase);

        // Two ranks: both ranks' evidence stretches by the iteration's one
        // correction, the heaviest rank's, so power is unchanged.
        let (grouped, evidence) = priced_with_evidence(&timing, |timing| {
            iteration_ms(timing, &[prefill.clone(), decode.clone()])
        });
        let correction = model
            .estimate_forward_pass_detailed(
                &queries.map(|query| AicTimingModel::rank_metrics(&query)),
            )
            .unwrap()
            .correction_factor
            .unwrap();
        assert!(close(grouped, correction * natives[0].max(natives[1])));
        assert!(!close(correction, 1.0), "{correction}");
        let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        for (tuned, untuned) in [
            (&evidence.prefill, &untuned.prefill),
            (&evidence.decode, &untuned.decode),
        ] {
            assert!(near(tuned.latency_ms, correction * untuned.latency_ms));
            assert!(near(energy(tuned), correction * energy(untuned)));
        }
    }

    #[test]
    fn op_level_iteration_values_record_the_named_evidence_bit_for_bit() {
        pyo3::prepare_freethreaded_python();
        let Some(mut timing) = packaged_op_level_timing(
            "op_level_iteration_values_record_the_named_evidence_bit_for_bit",
            true,
        ) else {
            return;
        };
        let model = Arc::clone(&timing.iteration.as_ref().unwrap().model);
        let mixed = |rank, batch, new, decode_batch, context| {
            mixed_work(
                prefill_work(rank, batch, new, 0),
                decode_work(rank, decode_batch, decode_batch * context, 1 << 20),
            )
        };
        // DP=3 passes mixing pure, mixed and idle ranks, each priced twice so
        // that cached iterations and rank evidence are recorded too.
        let groups = [
            vec![
                prefill_work(0, 4, 4096, 1024),
                decode_work(1, 32, 32 * 1025, 1 << 20),
                mixed(2, 1, 512, 8, 1025),
            ],
            vec![
                decode_work(0, 4, 4 * 1025, 1 << 20),
                RankWork::idle(1),
                decode_work(2, 32, 32 * 4097, 1 << 20),
            ],
            vec![
                mixed(0, 2, 300, 3, 2049),
                prefill_work(1, 1, 1000, 0),
                RankWork::idle(2),
            ],
            vec![
                decode_work(0, 16, 16 * 513, 1 << 20),
                mixed(1, 1, 128, 1, 1025),
                decode_work(2, 2, 2 * 700, 1 << 20),
            ],
        ];
        for capture in [false, true] {
            timing.diagnostic_model = capture.then(|| Arc::clone(&model));
            timing
                .iteration
                .as_ref()
                .unwrap()
                .cache
                .as_ref()
                .unwrap()
                .clear();
            let timing = &timing;
            let pricing = timing.iteration.as_ref().unwrap();
            // Reference: the detailed estimate's named operations, recorded
            // by the provider accumulator and by the checked accumulator.
            let mut named = TimingEvidenceAccumulator::default();
            let mut checked = TimingEvidenceSummary::default();
            timing.reset_evidence().unwrap();
            for ranks in groups.iter().chain(&groups) {
                let total_ms = iteration_ms(timing, ranks);
                let queries: Vec<_> = ranks
                    .iter()
                    .map(|rank| {
                        AicTimingModel::rank_query(pricing.dispatch, rank)
                            .unwrap()
                            .unwrap()
                    })
                    .collect();
                let metrics: Vec<_> = queries.iter().map(AicTimingModel::rank_metrics).collect();
                let estimate = model.estimate_forward_pass_detailed(&metrics).unwrap();
                assert_eq!(total_ms.to_bits(), estimate.latency_ms.unwrap().to_bits());
                for (phase, prefill) in timing
                    .rank_operation_evidence(pricing, &estimate, &queries)
                    .unwrap()
                {
                    named.record(&phase, prefill).unwrap();
                    let target = if prefill {
                        &mut checked.prefill
                    } else {
                        &mut checked.decode
                    };
                    target
                        .try_accumulate(phase.to_phase().into_owned())
                        .unwrap();
                }
            }
            let actual = timing.evidence_summary().unwrap();
            let named = named.snapshot();
            for (actual, named, checked) in [
                (&actual.prefill, &named.prefill, &checked.prefill),
                (&actual.decode, &named.decode, &checked.decode),
            ] {
                crate::engine::assert_phase_bits(actual, named);
                crate::engine::assert_phase_bits(actual, checked);
                assert!(actual.energy_wms.is_some());
                assert!(
                    actual
                        .operations
                        .iter()
                        .all(|operation| operation.details.is_some() == capture)
                );
            }
            if capture {
                continue;
            }
            // Mixed ranks add shared operations to decode, which pure decode
            // ranks lack; both kinds still update their operations in place.
            let pure_decode = model
                .estimate_native_iteration(
                    &[AicTimingModel::rank_metrics(&[0, 0, 0, 4, 4 * 1025])],
                    RankDetail::Values,
                )
                .unwrap()
                .unwrap();
            let pure_decode = &pure_decode.ranks[0].operations.as_ref().unwrap().names;
            let decode_names: Vec<_> = actual.decode.operations.iter().map(|op| &op.name).collect();
            assert!(pure_decode.iter().all(|name| decode_names.contains(&name)));
            assert!(decode_names.len() > pure_decode.len(), "{decode_names:?}");
            let (prefill_layouts, decode_layouts) = timing.evidence.lock().unwrap().slot_layouts();
            assert!(prefill_layouts >= 2, "{prefill_layouts}");
            assert!(decode_layouts >= 2, "{decode_layouts}");
        }
    }

    #[test]
    fn sd_round_iteration_prices_a_mixed_rank_as_one_forward() {
        pyo3::prepare_freethreaded_python();
        let tmp = tempfile::tempdir().unwrap();
        let model = Arc::new(ForwardPassPerfModel::from_engine(
            Arc::new(crate::perfmodel::engine::runtime::tests::full_round_engine(
                tmp.path(),
            )),
            Default::default(),
        ));
        let (mut phase, mut iteration) = fpm_timing_pair(&model);
        assert_eq!(
            iteration.iteration.as_ref().unwrap().query,
            IterationQuery::Native(IterationDispatch::WholeModel { sd_round: true })
        );
        // `full_round_spec`'s hand rows: the B1 prefill of 256 tokens after
        // 128 KV is 13 ms and the complete K3 decode round of one request at
        // 4096 past KV is 7 ms. The request's 4097-token context includes its
        // current token.
        let prefill = PrefillTotals {
            batch_size: 1,
            new_tokens: 256,
            prefix_tokens: 128,
        };
        let mixed = [mixed_work(
            prefill_work(0, 1, 256, 128),
            decode_work(0, 1, 4097, 1 << 20),
        )];
        for capture in [false, true] {
            if capture {
                phase.diagnostic_model = Some(Arc::clone(&model));
                iteration.diagnostic_model = Some(Arc::clone(&model));
            }
            // Phase pricing: the profile has no mix-step decode share, so the
            // scheduler prices a prefill pass and a separate decode round.
            assert_eq!(
                phase
                    .predict_mixed_pass_ms(prefill, 1, 4097, 4097, 1 << 20)
                    .unwrap(),
                None
            );
            let separate = phase.predict_prefill_totals_ms(1, 256, 128).unwrap()
                + phase.predict_decode_ms(1, 4097, 4097, 1 << 20).unwrap();
            assert_eq!(separate, 20.0);
            // Untuned, the one forward costs the same total. A tuned model
            // corrects it as the mixed iteration it is, where phase pricing
            // took each phase's own correction.
            assert_eq!(iteration_ms(&iteration, &mixed), separate);
        }
        // Attention DP stays outside the measured identity.
        assert_eq!(
            iteration
                .predict_iteration_ms(&[mixed[0].clone(), RankWork::idle(1)])
                .unwrap(),
            None
        );
    }

    #[test]
    fn bounded_replay_rejects_heterogeneous_actual_prefill_geometry() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let mut timing = python_timing_model(engine, false);
        timing.decoder_replay = true;
        for geometry in [vec![], vec![(3, 1536)], vec![(129, 128), (129, 128)]] {
            timing.validate_prefill_batch(&geometry).unwrap();
        }
        for geometry in [vec![(127, 128), (129, 128)], vec![(3, 128), (3, 1536)]] {
            let error = timing.validate_prefill_batch(&geometry).unwrap_err();
            assert!(error.to_string().contains("heterogeneous prefill"));
            timing.decoder_replay = false;
            timing.validate_prefill_batch(&geometry).unwrap();
            timing.decoder_replay = true;
        }
    }

    #[test]
    fn op_level_timing_exposes_typed_python_evidence() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| {
            Py::new(py, PerOpEvidenceProbe::default())
                .unwrap()
                .into_any()
        });
        let timing = python_timing_model(engine, false);

        assert_eq!(timing.predict_prefill_ms(2, 128, 0).unwrap(), 10.0);
        assert_eq!(timing.predict_decode_ms(2, 258, 128, 1024).unwrap(), 4.0);

        let evidence = timing.evidence_summary().unwrap();
        assert_eq!(evidence.prefill.energy_wms, Some(3_200.0));
        assert_eq!(evidence.prefill.latency_ms, 10.0);
        assert_eq!(evidence.prefill.covered_latency_ms, 8.0);
        assert_eq!(evidence.prefill.coverage(), 0.8);
        assert_eq!(evidence.prefill.source, Some(TimingEvidenceSource::Mixed));
        assert_eq!(evidence.prefill.operations.len(), 2);
        assert_eq!(evidence.prefill.operations[1].energy_wms, None);
        assert_eq!(
            evidence.prefill.operations[1].source,
            TimingEvidenceSource::Empirical
        );
        assert_eq!(evidence.decode.energy_wms, Some(1_600.0));
        assert_eq!(evidence.decode.coverage(), 1.0);
    }

    #[test]
    fn repeated_shapes_reuse_provider_evidence_and_accumulate_every_step() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, PerOpEvidenceProbe::default()).unwrap());
        let timing = python_timing_model(
            Python::with_gil(|py| engine.clone_ref(py).into_any()),
            false,
        );
        for _ in 0..1_000 {
            assert_eq!(timing.predict_decode_ms(2, 258, 128, 1024).unwrap(), 4.0);
        }
        Python::with_gil(|py| {
            assert_eq!(
                engine
                    .borrow(py)
                    .calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                1
            )
        });
        assert_eq!(
            timing.evidence_summary().unwrap().decode.energy_wms,
            Some(1_600_000.0)
        );
        let cached = timing.phase_cache.get(&(2, 127, 2, 0, false)).unwrap();
        let again = timing.phase_cache.get(&(2, 127, 2, 0, false)).unwrap();
        assert!(Arc::ptr_eq(&cached, &again));
        assert_eq!(cached.to_phase().energy_wms, Some(1_600.0));
        // Distinct coordinates must never reuse the preceding shape's result.
        timing.predict_decode_ms(2, 260, 129, 1024).unwrap();
        Python::with_gil(|py| {
            assert_eq!(
                engine
                    .borrow(py)
                    .calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                2
            )
        });
    }

    #[test]
    fn aic_measurement_reset_clears_phase_provenance_without_clearing_shape_cache() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, PerOpEvidenceProbe::default()).unwrap());
        let timing = python_timing_model(
            Python::with_gil(|py| engine.clone_ref(py).into_any()),
            false,
        );
        timing.predict_prefill_ms(2, 128, 0).unwrap();
        timing.predict_decode_ms(2, 258, 128, 1024).unwrap();
        let before = timing.evidence_summary().unwrap();
        assert_eq!(before.prefill.source, Some(TimingEvidenceSource::Mixed));
        assert!(replay_power_stats(&before).unwrap().coverage() < 1.0);

        timing.reset_evidence().unwrap();
        assert_eq!(
            timing.evidence_summary(),
            Some(TimingEvidenceSummary::default())
        );
        timing.predict_decode_ms(2, 258, 128, 1024).unwrap();
        let after = timing.evidence_summary().unwrap();
        assert_eq!(after.prefill, TimingPhaseEvidence::default());
        assert_eq!(after.decode, before.decode);
        assert_eq!(replay_power_stats(&after).unwrap().coverage(), 1.0);
        assert_eq!(replay_power_stats(&after).unwrap().power_w(), Some(400.0));
        Python::with_gil(|py| {
            assert_eq!(
                engine
                    .borrow(py)
                    .calls
                    .load(std::sync::atomic::Ordering::Relaxed),
                2
            );
        });
        timing.reset_evidence().unwrap();
        assert_eq!(
            timing.evidence_summary(),
            Some(TimingEvidenceSummary::default())
        );
    }

    #[test]
    fn empty_native_evidence_and_diagnostics_reject_nonzero_work() {
        use crate::perfmodel::engine::{Engine, spec::EngineSpec};
        use crate::perfmodel::fpm::ForwardPassPerfOptions;
        pyo3::prepare_freethreaded_python();
        let db = crate::perf_database::PerfDatabase::load(
            &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems"),
            "h200_sxm",
            "sglang",
            "0.5.6.post2",
        )
        .unwrap();
        let config = serde_json::from_value(serde_json::json!({
            "schema_version": crate::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "empty-fixture", "system_name": "h200_sxm",
            "backend": "sglang", "backend_version": "0.5.6.post2",
            "tp_size": 1, "pp_size": 1
        }))
        .unwrap();
        let native =
            Arc::new(Engine::build(EngineSpec::new(config, vec![], vec![]), Arc::new(db)).unwrap());
        let engine = Python::with_gil(|py| {
            Py::new(py, PerOpEvidenceProbe::default())
                .unwrap()
                .into_any()
        });
        let mut timing = python_timing_model(engine, false);
        timing.phase_provider = AicPhaseProvider::Native(Arc::clone(&native));
        timing.diagnostic_model = Some(Arc::new(ForwardPassPerfModel::from_engine(
            native,
            ForwardPassPerfOptions::default(),
        )));
        for kind in ["diagnostics", "evidence"] {
            for mode in ["static_ctx", "static_gen"] {
                let error = timing
                    .predict_phase_evidence(1, 128, 2, 0, mode)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains(&format!("AIC {mode} returned empty operation {kind}"))
                );
                assert_eq!(
                    timing
                        .predict_phase_evidence(0, 128, 2, 0, mode)
                        .unwrap()
                        .latency_ms(),
                    0.0
                );
            }
            assert_eq!(
                timing
                    .predict_phase_evidence(1, 128, 2, 128, "static_ctx")
                    .unwrap()
                    .latency_ms(),
                0.0
            );
            assert_eq!(
                timing.evidence_summary(),
                Some(TimingEvidenceSummary::default())
            );
            timing.diagnostic_model = None;
        }
    }

    #[test]
    fn empty_provider_evidence_rejects_nonzero_work() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, PerOpEvidenceProbe::default()).unwrap());
        let timing = python_timing_model(
            Python::with_gil(|py| engine.clone_ref(py).into_any()),
            false,
        );
        assert!(timing.predict_prefill_ms(99, 128, 0).is_err());
        assert!(timing.predict_decode_ms(99, 12800, 128, 16384).is_err());
        let empty = timing
            .predict_phase_evidence(0, 128, 1, 0, "static_ctx")
            .unwrap();
        for seeded in [false, true] {
            if seeded {
                timing.predict_prefill_ms(2, 128, 0).unwrap();
                timing.predict_decode_ms(2, 258, 128, 1024).unwrap();
            }
            let before = timing.evidence_summary();
            let calls = Python::with_gil(|py| {
                engine
                    .borrow(py)
                    .calls
                    .load(std::sync::atomic::Ordering::Relaxed)
            });
            for _ in 0..2 {
                assert_eq!(timing.predict_prefill_ms(0, 128, 0).unwrap(), 0.0);
                assert_eq!(timing.predict_decode_ms(0, 0, 128, 16384).unwrap(), 0.0);
                assert_eq!(timing.predict_prefill_ms(99, 128, 128).unwrap(), 0.0);
                assert!(Arc::ptr_eq(
                    &empty,
                    &timing
                        .predict_phase_evidence(99, 128, 1, 128, "static_ctx")
                        .unwrap()
                ));
                assert_eq!(timing.evidence_summary(), before);
            }
            Python::with_gil(|py| {
                assert_eq!(
                    engine
                        .borrow(py)
                        .calls
                        .load(std::sync::atomic::Ordering::Relaxed),
                    calls
                )
            });
        }
    }

    #[test]
    fn native_evidence_builder_matches_checked_values_and_errors() {
        let row = |name: &str, latency, energy, source| (name.to_owned(), latency, energy, source);
        let mut cases = vec![Vec::new(), vec![row("zero", -0.0, -0.0, "unknown")]];
        for count in [1, 24, 32] {
            cases.push(
                (0..count)
                    .map(|i| {
                        row(
                            &format!("op-{i}"),
                            [0.1, 1e8, -0.0, f64::MIN_POSITIVE][i % 4],
                            [0.0, 1e9, -0.0, 0.1][i % 4],
                            ["silicon", "empirical", "sol", "estimated", "mixed", "other"][i % 6],
                        )
                    })
                    .collect(),
            );
        }
        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            cases.push(vec![row("invalid", invalid, 1.0, "silicon")]);
            cases.push(vec![row("invalid", 1.0, invalid, "silicon")]);
            // The raw-energy error precedes both name and latency errors.
            cases.push(vec![row("", invalid, invalid, "other")]);
        }
        cases.push(vec![row("", 1.0, 1.0, "silicon")]);
        for (latency, energy) in [(f64::MAX, 0.0), (1.0, f64::MAX)] {
            let overflow = vec![
                row("first", latency, energy, "silicon"),
                row("second", latency, energy, "silicon"),
            ];
            cases.push(overflow.clone());
            let mut later_error = overflow;
            later_error.push(row("later", f64::NAN, 1.0, "silicon"));
            cases.push(later_error);
        }
        let assert_bits = |actual: &TimingPhaseEvidence, expected: &TimingPhaseEvidence| {
            assert_eq!(actual, expected);
            let bits = |phase: &TimingPhaseEvidence| {
                std::iter::once((phase.latency_ms, phase.energy_wms, phase.covered_latency_ms))
                    .chain(
                        phase
                            .operations
                            .iter()
                            .map(|op| (op.latency_ms, op.energy_wms, op.covered_latency_ms)),
                    )
                    .map(|(latency, energy, covered)| {
                        (
                            latency.to_bits(),
                            energy.map(f64::to_bits),
                            covered.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(actual), bits(expected));
            assert_eq!(actual.coverage().to_bits(), expected.coverage().to_bits());
            let a = phase_power_stats(actual).unwrap();
            let b = phase_power_stats(expected).unwrap();
            assert_eq!(a.power_w.map(f64::to_bits), b.power_w.map(f64::to_bits));
            assert_eq!(a.coverage.to_bits(), b.coverage.to_bits());
        };
        pyo3::prepare_freethreaded_python();
        let mut timing = Python::with_gil(|py| python_timing_model(py.None(), false));
        timing.phase_provider = AicPhaseProvider::NativeEntries(vec![row("seed", 1.0, 2.0, "sol")]);
        timing.predict_decode_ms(1, 65, 64, 1024).unwrap();
        let cached = timing.phase_cache.get(&(1, 63, 2, 0, false)).unwrap();
        let before = timing.evidence_summary().unwrap();
        timing.phase_provider = AicPhaseProvider::NativeEntries(Vec::new());
        for _ in 0..2 {
            let error = timing.predict_decode_ms(1, 129, 128, 1024).unwrap_err();
            assert_eq!(
                error.to_string(),
                "AIC static_gen returned empty operation evidence for nonzero work"
            );
            assert!(timing.phase_cache.get(&(1, 127, 2, 0, false)).is_none());
            assert!(Arc::ptr_eq(
                &cached,
                &timing.phase_cache.get(&(1, 63, 2, 0, false)).unwrap()
            ));
            let after = timing.evidence_summary().unwrap();
            assert_bits(&after.prefill, &before.prefill);
            assert_bits(&after.decode, &before.decode);
        }
        for entries in cases {
            let checked = phase_evidence_from_entries(
                entries
                    .iter()
                    .map(|(name, latency, energy, source)| {
                        (name.clone(), *latency, *energy, (*source).to_owned())
                    })
                    .collect(),
            );
            let native = phase_evidence_from_native_entries(entries.clone());
            match (native, checked) {
                (Ok(native), Ok(checked)) => assert_bits(&native.to_phase(), &checked.to_phase()),
                (Err(native), Err(checked)) => {
                    assert_eq!(native.to_string(), checked.to_string());
                    timing.phase_provider = AicPhaseProvider::NativeEntries(entries);
                    for _ in 0..2 {
                        let error = timing.predict_decode_ms(1, 129, 128, 1024).unwrap_err();
                        assert_eq!(error.to_string(), checked.to_string());
                        assert!(timing.phase_cache.get(&(1, 127, 2, 0, false)).is_none());
                        assert!(Arc::ptr_eq(
                            &cached,
                            &timing.phase_cache.get(&(1, 63, 2, 0, false)).unwrap()
                        ));
                        let after = timing.evidence_summary().unwrap();
                        assert_bits(&after.prefill, &before.prefill);
                        assert_bits(&after.decode, &before.decode);
                    }
                }
                mismatch => panic!("native builder mismatch: {mismatch:?}"),
            }
        }
    }

    #[test]
    fn python_evidence_rejects_invalid_energy_before_missing_value_conversion() {
        let error =
            phase_evidence_from_entries(vec![("bad".into(), 1.0, f64::NAN, "silicon".into())])
                .unwrap_err();
        assert!(error.to_string().contains("invalid energy"));
    }

    #[test]
    fn json_bindings_share_report_and_retain_request_correlation() {
        let mut spec = ReplaySpec {
            version: 1,
            encoder: None,
            topology: ReplayTopology::Aggregated {
                workers: WorkerPoolSpec::default(),
            },
            engine: serde_json::to_value(ReplayEngineConfig {
                rank: EngineConfig {
                    num_gpu_blocks: 16,
                    block_size: 4,
                    max_num_seqs: 4,
                    max_num_batched_tokens: 64,
                    timing_model: TimingModelConfig::Fixed {
                        prefill_ms: 1.0,
                        decode_ms: 1.0,
                    },
                    ..EngineConfig::default()
                },
                ..ReplayEngineConfig::default()
            })
            .unwrap(),
            adapters: ReplayAdapters {
                placement: ProviderSpec::round_robin(),
                scaling: ProviderSpec::no_scaling(),
            },
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests: vec![ReplayRequest {
                id: "authored-id".into(),
                arrival_time_ms: 0.0,
                input_tokens: 4,
                input_token_ids: Some(vec![1, 2, 3, 4]),
                output_tokens: 1,
                output_token_ids: None,
                dp_rank: None,
                prefill_dp_rank: None,
                kv_hints: None,
                session_id: Some("session-a".into()),
                turn_index: Some(2),
                metadata: serde_json::json!({"caller_tag": "binding"}),
            }],
        };

        let payload = serde_json::to_string(&spec).unwrap();
        let output = execute_json(&payload, false).unwrap();
        let report: serde_json::Value = serde_json::from_str(&output).unwrap();
        let record = &report["per_request"][0];
        assert_eq!(record["request_id"], "authored-id");
        assert_eq!(record["session_id"], "session-a");
        assert_eq!(record["turn_index"], 2);
        assert_eq!(record["metadata"]["caller_tag"], "binding");

        // Streamed records serialize exactly like the whole report `Value`.
        let replayed = Replayer::new(spec.clone(), ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        let mut summary = serde_json::to_value(&replayed).unwrap();
        summary["per_request"] = serde_json::Value::Null;
        let mut whole = summary.clone();
        whole["per_request"] = serde_json::to_value(&replayed.per_request).unwrap();
        let streamed = ReportWithRecords {
            report: &summary,
            records: &replayed.per_request,
        };
        assert_eq!(
            serde_json::to_string(&streamed).unwrap(),
            serde_json::to_string(&whole).unwrap()
        );

        let captured: serde_json::Value =
            serde_json::from_str(&execute_json(&payload, true).unwrap()).unwrap();
        assert_eq!(captured["report"]["completed_requests"], 1);
        assert_eq!(
            captured["report"]["per_request"][0]["request_id"],
            "authored-id"
        );
        assert_eq!(
            captured["artifacts"]["requests"].as_array().unwrap().len(),
            1
        );
        assert_eq!(
            captured["artifacts"]["requests"][0]["request_id"],
            captured["report"]["per_request"][0]["uuid"]
        );

        spec.requests.clear();
        for record_per_request in [false, true] {
            spec.record_per_request = record_per_request;
            let payload = serde_json::to_string(&spec).unwrap();
            for capture_artifacts in [false, true] {
                let output: serde_json::Value =
                    serde_json::from_str(&execute_json(&payload, capture_artifacts).unwrap())
                        .unwrap();
                let report = if capture_artifacts {
                    &output["report"]
                } else {
                    &output
                };
                assert_eq!(report["completed_requests"], 0);
                assert!(report.get("agentic_phases").is_none());
                if record_per_request {
                    assert_eq!(report["per_request"], serde_json::json!([]));
                } else {
                    assert!(report.get("per_request").is_none());
                }
            }
        }
    }
}
