// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-only PyO3 boundary for one materialized AISimulate replay execution.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::engine::{
    Backend, EngineConfig, TimingEvidenceAccumulator, TimingEvidenceSource, TimingEvidenceSummary,
    TimingModel, TimingModelConfig, TimingOperationEvidence, TimingPhaseEvidence,
    ValidatedTimingPhase,
};
use crate::perfmodel::engine::{Engine as PerfEngine, RuntimeConfig};
use crate::replay::{
    POWER_DATA_COVERAGE_THRESHOLD, ReplayArtifactKvEventVisibility, ReplayArtifacts,
    ReplayEngineConfig, ReplayEngineFactory, ReplayOperationPowerDiagnostics,
    ReplayPhasePowerDiagnostics, ReplayPowerDiagnostics, ReplayRoleConfig, ReplayRuntimeInput,
    ReplaySpec, ReplayTopology, Replayer, TracePowerStats,
    loadgen::{
        AgenticSnapshotOptions, ArrivalSpec, DelaySpec, DynamoRequestTrace, LengthSpec,
        SyntheticTraceSpec, Trace, ValidatedAgenticGraph, WekaImportOptions,
        WekaNestedTimestampBasis, WekaResolvedTimestampBasis, WorkloadDriver,
        load_agentic_mooncake, load_weka_agentic_graph_with_options,
    },
};
use crate::{
    EstimationMode, EstimatorConfig, ForwardPassFallbackPolicy, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassWorkerType,
};
use anyhow::{Context, Result, anyhow, ensure};
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
    },
    Legacy(ReplaySpec),
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
    isl: Option<usize>,
    #[serde(default)]
    osl: Option<usize>,
    #[serde(default)]
    cached_prefix_tokens: Option<usize>,
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

fn validate_public_agentic_engine(input: &ReplayRuntimeInput, rank: &EngineConfig) -> Result<()> {
    let ReplayRuntimeInput::Workload(driver) = input else {
        return Ok(());
    };
    if !driver.is_agentic() {
        return Ok(());
    }
    ensure!(
        matches!(rank.backend, Backend::Vllm | Backend::Sglang),
        "agentic replay supports only vLLM and SGLang backends"
    );
    ensure!(
        rank.native_host_offload.is_none(),
        "agentic replay requires HBM-only KV cache; host offload is unsupported"
    );
    ensure!(
        rank.aic_nextn.is_none(),
        "agentic replay requires speculative decoding disabled"
    );
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AicTimingConfig {
    model: String,
    backend: String,
    system: String,
    #[serde(default = "one", alias = "tp_size")]
    tp: u32,
    #[serde(default)]
    backend_version: Option<String>,
    #[serde(default = "one")]
    pp: u32,
    #[serde(default = "one")]
    attention_dp: u32,
    #[serde(default)]
    moe_tp_size: Option<u32>,
    #[serde(default)]
    moe_ep_size: Option<u32>,
    /// Prefill context parallelism (SGLang attn-cp / vLLM PCP): folds into the
    /// attention width like attention_dp. `None` means 1.
    #[serde(default)]
    cp_size: Option<u32>,
    /// Decode context parallelism (vLLM `-dcp` / SGLang `--dcp-size`): stripes
    /// the decode KV inside the attention group, so it does NOT widen the
    /// topology. `None` means 1.
    #[serde(default)]
    dcp_size: Option<u32>,
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
    fpm_parquet_path: Option<String>,
    #[serde(default)]
    decoder_replay: bool,
    #[serde(default)]
    worker_type: Option<ForwardPassWorkerType>,
    #[serde(default)]
    estimation_mode: Option<EstimationMode>,
    #[serde(default)]
    fallback_policy: ForwardPassFallbackPolicy,
    #[serde(default)]
    estimator_config: EstimatorConfig,
    #[serde(default)]
    database_mode: crate::DatabaseMode,
    #[serde(default)]
    transfer_policy: Option<Vec<String>>,
    #[serde(default)]
    systems_paths: Vec<PathBuf>,
    #[serde(default)]
    attention_backend: Option<String>,
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
        let mut estimator_config = self.estimator_config.clone();
        if let Some(path) = &self.fpm_parquet_path {
            crate::config::validate_fpm_parquet_path(
                Some(std::path::Path::new(path)),
                mode == EstimationMode::FpmInterpolation,
            )?;
            let configured = &mut estimator_config.fpm_interpolation.fpm_parquet_path;
            ensure!(
                configured
                    .as_ref()
                    .is_none_or(|existing| existing == std::path::Path::new(path)),
                "conflicting fpm_parquet_path and estimator_config.fpm_interpolation.fpm_parquet_path"
            );
            *configured = Some(path.into());
        }
        Ok(ForwardPassPerfModelConfig {
            model: self.model.clone(),
            system: self.system.clone(),
            backend: serde_json::from_value(serde_json::Value::String(self.backend.clone()))?,
            backend_version: self.backend_version.clone(),
            worker_type,
            tp: self.tp,
            pp: self.pp,
            attention_dp: self.attention_dp,
            moe_tp_size: self.moe_tp_size,
            moe_ep_size: self.moe_ep_size,
            cp_size: self.cp_size,
            dcp_size: self.dcp_size,
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
            database_mode: self.database_mode,
            transfer_policy: self.transfer_policy.clone(),
            systems_paths: roots,
            attention_backend: self.attention_backend.clone(),
            moe_backend: self.moe_backend.clone(),
            enable_eplb: self.enable_eplb,
            wideep_num_slots: self.wideep_num_slots,
            enable_shared_layer: self.enable_shared_layer,
            strict_provenance: self.strict_provenance,
        })
    }

    fn speculative_depth(&self) -> Result<u32> {
        let Some(speculation) = &self.speculation else {
            return Ok(self.nextn);
        };
        ensure!(
            self.nextn == 0,
            "ngram speculation cannot be combined with nextn"
        );
        ensure!(
            self.backend == "vllm",
            "ngram speculation requires backend=vllm"
        );
        ensure!(
            self.forward_model.as_deref().unwrap_or("op_level") == "op_level",
            "ngram speculation requires op_level timing"
        );
        let depth = speculation.num_speculative_tokens();
        ensure!(
            (1..=5).contains(&depth),
            "ngram num_speculative_tokens must be in 1..=5"
        );
        Ok(depth)
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
                && self.dcp_size != Some(0),
            "AIC timing parallel sizes tp, pp, attention_dp, moe_tp_size, \
             moe_ep_size, cp_size, and dcp_size must be positive"
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
                u64::from(self.tp) * u64::from(self.attention_dp) * cp
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
    /// Keep the original bridge as an independent test reference and probe seam.
    #[cfg(test)]
    Python,
    /// Inject native rows to test construction failures before cache publication.
    #[cfg(test)]
    NativeEntries(Vec<crate::perfmodel::engine::PerOpValue>),
}

struct AicTimingModel {
    engine: Py<PyAny>,
    diagnostic_model: Option<ForwardPassPerfModel>,
    decoder_replay: bool,
    phase_provider: AicPhaseProvider,
    use_fpm_decode_totals: bool,
    fpm_decode_kv_ceiling: Option<u32>,
    evidence: Mutex<TimingEvidenceAccumulator>,
    phase_cache: quick_cache::sync::Cache<PhaseEvidenceKey, Arc<ValidatedTimingPhase>>,
}

impl AicTimingModel {
    fn build(config: &mut AicTimingConfig, worker_type: ForwardPassWorkerType) -> Result<Self> {
        config.validate_parallel_shape()?;
        config.resolved_memory_fraction()?;
        let model = ForwardPassPerfModel::best_available(config.estimator_request(worker_type)?)
            .context("AIC timing provider could not construct the requested estimator")?;
        let provenance = model
            .provenance()
            .context("canonical estimator omitted construction provenance")?;
        config.backend_version = provenance.config.backend_version.clone();
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
        config.estimator_config = provenance.config.estimator_config.clone();
        let use_fpm_decode_totals =
            provenance.selected_estimation_mode == EstimationMode::FpmInterpolation;
        let native = model.native_engine().context(
            "AIC regression estimator is not ready: offline replay requires trained observations or a native estimator"
        )?;
        let phase_provider = AicPhaseProvider::Native(Arc::clone(&native));
        let (engine, fpm_decode_kv_ceiling) = Python::with_gil(|py| -> PyResult<_> {
            let engine = Py::new(py, crate::AicEngine::from_shared_engine(native))?.into_any();
            let ceiling = if use_fpm_decode_totals {
                engine
                    .bind(py)
                    .call_method0("fpm_decode_kv_ceiling")?
                    .extract::<Option<u32>>()?
            } else {
                None
            };
            Ok((engine, ceiling))
        })?;
        Ok(Self {
            engine,
            diagnostic_model: Some(model),
            decoder_replay: config.decoder_replay,
            phase_provider,
            use_fpm_decode_totals,
            fpm_decode_kv_ceiling,
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            phase_cache: quick_cache::sync::Cache::new(128),
        })
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
            let phase = Arc::new(ValidatedTimingPhase::from_operations(operations)?);
            self.phase_cache.insert(key, Arc::clone(&phase));
            return Ok(phase);
        }
        let phase = match &self.phase_provider {
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
                let (context, generation) = crate::py::parse_mode(mode)
                    .and_then(|mode| {
                        engine.reset_provenance();
                        engine
                            .run_static_per_op(
                                &runtime,
                                mode,
                                crate::perfmodel::engine::DEFAULT_STATIC_STRIDE,
                            )
                            .map_err(crate::py::aic_to_py)
                    })
                    .map_err(|error| anyhow!("AIC {mode} evidence prediction failed: {error}"))?;
                let entries = if prefill { context } else { generation };
                ensure!(
                    !entries.is_empty(),
                    "AIC {mode} returned empty operation evidence for nonzero work"
                );
                phase_evidence_from_native_entries(entries)?
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
}

fn phase_evidence_from_native_entries(
    entries: Vec<crate::perfmodel::engine::PerOpValue>,
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
                TimingEvidenceSource::from_native(source),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ValidatedTimingPhase::from_name_folded_operations(operations)
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
    fn prefill_batch_validation_can_fail(&self) -> bool {
        self.decoder_replay
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
        let batch_size = checked_u32(batch_size, "prefill batch size")?;
        let mean_isl = checked_u32(mean_isl, "mean input length")?;
        let mean_prefix = checked_u32(mean_prefix, "mean prefix length")?;
        if !self.use_fpm_decode_totals {
            let evidence =
                self.predict_phase_evidence(batch_size, mean_isl, 1, mean_prefix, "static_ctx")?;
            let latency_ms = evidence.as_phase().latency_ms;
            self.record_evidence(&evidence, true)?;
            return Ok(latency_ms);
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
        .map_err(|error| anyhow!("AIC prefill prediction failed: {error}"))
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        if self.use_fpm_decode_totals {
            let total_past_kv_tokens = active_kv_tokens
                .checked_sub(batch_size)
                .context("active decode tokens must include one current token per request")?
                .min(total_kv_tokens);
            let batch_size = checked_u32(batch_size, "decode batch size")?;
            let total_past_kv_tokens = checked_u32(total_past_kv_tokens, "total past KV tokens")?;
            return Python::with_gil(|py| {
                self.engine
                    .bind(py)
                    .call_method1(
                        "predict_decode_latency_total",
                        (batch_size, total_past_kv_tokens),
                    )?
                    .extract::<f64>()
            })
            .map_err(|error| anyhow!("AIC decode prediction failed: {error}"));
        }

        let batch_size = checked_u32(batch_size, "decode batch size")?;
        // Both scheduler implementations include the current input token in
        // their sequence length before sampling the next token. The op API's
        // isl is past KV; osl=2 adds the current token exactly once.
        let mean_past_kv = mean_context_length
            .checked_sub(1)
            .context("mean decode context must include the current input token")?;
        let mean_past_kv = checked_u32(mean_past_kv, "mean past KV length")?;
        let evidence = self.predict_phase_evidence(batch_size, mean_past_kv, 2, 0, "static_gen")?;
        let latency_ms = evidence.as_phase().latency_ms;
        self.record_evidence(&evidence, false)?;
        Ok(latency_ms)
    }

    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        if self.use_fpm_decode_totals {
            return None;
        }
        self.evidence
            .lock()
            .ok()
            .map(|evidence| evidence.snapshot())
    }

    fn reset_evidence(&self) -> Result<()> {
        *self
            .evidence
            .lock()
            .map_err(|_| anyhow!("AIC timing evidence accumulator was poisoned"))? =
            TimingEvidenceAccumulator::default();
        Ok(())
    }
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} {value} exceeds AIC's u32 limit"))
}

fn estimate_aic_num_gpu_blocks(config: &AicTimingConfig, role: &ReplayRoleConfig) -> Result<usize> {
    let (memory_fraction_kind, memory_fraction_value) = config.resolved_memory_fraction()?;
    Python::with_gil(|py| -> PyResult<usize> {
        let memory = PyModule::import(py, "aisimulate_core.sdk.memory")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backend_version", config.resolved_backend_version())?;
        kwargs.set_item("scheduler_block_size", role.rank.block_size)?;
        kwargs.set_item("max_num_tokens", role.rank.max_num_batched_tokens)?;
        kwargs.set_item("max_batch_size", role.rank.max_num_seqs)?;
        kwargs.set_item("memory_fraction_kind", memory_fraction_kind)?;
        kwargs.set_item("memory_fraction_value", memory_fraction_value)?;
        kwargs.set_item("tp_size", config.tp)?;
        kwargs.set_item("pp_size", config.pp)?;
        kwargs.set_item("attention_dp_size", config.attention_dp)?;
        kwargs.set_item("moe_tp_size", config.moe_tp_size)?;
        kwargs.set_item("moe_ep_size", config.moe_ep_size)?;
        kwargs.set_item("cp_size", config.cp_size.unwrap_or(1))?;
        kwargs.set_item("dcp_size", config.dcp_size.unwrap_or(1))?;
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
            "cuda_graph_reserved_bytes",
            config.cuda_graph_reserved_bytes,
        )?;
        // Capacity intentionally omits NextN until AIC's Eagle memory model no
        // longer returns negative KV capacity. Timing compilation still uses it.
        kwargs.set_item("systems_path", config.systems_path.as_deref())?;
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

fn materialize_aic_capacity(
    config: &AicTimingConfig,
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
    estimate: impl FnOnce(&AicTimingConfig, &ReplayRoleConfig) -> Result<usize>,
) -> Result<()> {
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
    ensure!(
        config.tp == role.tensor_parallel_size,
        "AIC tp={} does not match engine tensor_parallel_size={}",
        config.tp,
        role.tensor_parallel_size
    );
    ensure!(
        config.attention_dp == role.dp_size,
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
    if capacity_is_explicit || role.rank.state_cache.is_some() {
        return Ok(());
    }
    let blocks = estimate(config, role)?;
    ensure!(blocks > 0, "AIC estimated zero KV-cache blocks");
    role.rank.num_gpu_blocks = blocks;
    Ok(())
}

fn cap_role_capacity_to_fpm_decode_domain(
    role: &mut ReplayRoleConfig,
    decode_kv_ceiling: Option<u32>,
    capacity_is_explicit: bool,
) -> Result<()> {
    let Some(decode_kv_ceiling) = decode_kv_ceiling else {
        return Ok(());
    };
    if capacity_is_explicit || role.rank.state_cache.is_some() {
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

fn resolve_role_timing(
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
    worker_type: ForwardPassWorkerType,
    capture_performance_diagnostics: bool,
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
    let mut timing = AicTimingModel::build(&mut config, worker_type)?;
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
        capacity_is_explicit,
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
    let expected_tokens = isl
        .checked_add(osl / 2)
        .context("KV-load expected token count overflow")?;
    ensure!(
        expected_tokens > 0,
        "KV load requires positive token lengths"
    );
    let per_rank_tokens = role
        .rank
        .num_gpu_blocks
        .checked_mul(role.rank.block_size)
        .context("KV capacity overflow")?;
    let total_tokens = per_rank_tokens
        .checked_mul(role.dp_size as usize)
        .and_then(|value| value.checked_mul(replicas))
        .context("aggregate KV capacity overflow")?;
    let capacity = total_tokens / expected_tokens;
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
        let prepared = graph.prepare_snapshots(lanes, *options)?;
        if traffic.agentic_warmup {
            WorkloadDriver::new_agentic_warmup(prepared, engine_block_size, true, speedup)
        } else {
            WorkloadDriver::new_agentic_snapshots(prepared, engine_block_size, true, speedup)
        }
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
            let (trace, resolved_basis) = load_weka_agentic_graph_with_options(
                &paths[0],
                traffic.trace_block_size,
                WekaImportOptions {
                    nested_timestamp_basis: requested_basis,
                },
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
    let trace = Trace::synthetic(SyntheticTraceSpec {
        // A one-token trace block preserves prefixes that are not aligned to
        // the engine's scheduler block size. The driver still hashes them at
        // `engine_block_size` when it constructs replay requests.
        block_size: if cached_prefix_tokens == 0 {
            engine_block_size
        } else {
            1
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
        shared_prefix_ratio: traffic.shared_prefix_ratio.unwrap_or(0.0),
        num_prefix_groups: traffic.num_prefix_groups.unwrap_or(0),
        first_turn_arrivals: synthetic_arrivals(&traffic)?,
        inter_turn_delays: traffic
            .inter_turn_delay_ms
            .filter(|delay| *delay > 0.0)
            .map_or(DelaySpec::None, DelaySpec::ConstantMs),
        seed: 0,
        arrival_seed: traffic.arrival_seed.unwrap_or(42),
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
) -> crate::replay::ReplayResult<(crate::replay::ReplayReport, Option<ReplayArtifacts>)> {
    let replayer = match input {
        Some(input) => Replayer::new(spec, factory)?.with_runtime_input(input),
        None => Replayer::new(spec, factory)?,
    };
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
    let mut combined = TimingEvidenceSummary::default();
    for source in sources {
        let Some(summary) = source.timing.evidence_summary() else {
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
            "unavailable_reason": "selected timing provider does not export operation diagnostics (whole-model FPM and latency-only providers are unsupported)"});
    };
    let phases = [("prefill", &summary.prefill), ("decode", &summary.decode)].into_iter().map(|(name, phase)| {
        let mut operations = phase.operations.iter().map(|op| {
            let sol = op.details.as_ref().and_then(|d| d.sol.as_ref());
            serde_json::json!({"name": op.name, "latency_ms": op.latency_ms,
                "source": op.source.as_str(),
                "sol": sol,
                "sol_unavailable_reason": if sol.is_some() { None } else {
                    op.details.as_ref().and_then(|d| d.sol_unavailable_reason.as_deref()).or(Some("provider did not export SOL evidence"))
                },
                "latency_to_sol_ratio": sol.filter(|s| s.latency_ms > 0.0).map(|s| op.latency_ms / s.latency_ms).filter(|r| r.is_finite()),
                "fallbacks": op.details.as_ref().map(|d| &d.fallbacks),
            })
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
        ] {
            if let Some(value) = config.get(fields[0]).or_else(|| config.get(fields[1])) {
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

fn execute_json(payload: &str, capture_artifacts: bool) -> Result<String> {
    let (mut spec, mut traffic, capture_performance_diagnostics) =
        match serde_json::from_str(payload).context("invalid AISimulate execution ReplaySpec")? {
            ExecutionPayload::Configured {
                spec,
                traffic,
                capture_performance_diagnostics,
            } => (spec, traffic.map(|t| *t), capture_performance_diagnostics),
            ExecutionPayload::Legacy(spec) => (spec, None, false),
        };
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
    let expected_power_sources = match &spec.topology {
        ReplayTopology::Aggregated { .. } => 1,
        ReplayTopology::Disaggregated { .. } => 2,
    };
    let mut power_sources = Vec::with_capacity(expected_power_sources);

    let (mut report, artifacts, resolved_weka_timestamp_basis) = match spec.topology.clone() {
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
            )?;
            if let Some(timing) = timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &role.rank));
            }
            engine_config.dp_size = role.dp_size;
            engine_config.tensor_parallel_size = role.tensor_parallel_size;
            engine_config.num_gpu_blocks_is_explicit = role.num_gpu_blocks_is_explicit;
            engine_config.rank = role.rank;
            if let Some(traffic) = traffic.as_mut()
                && let ReplayTopology::Aggregated { workers } = &spec.topology
                && let Some(concurrency) = resolve_kv_capacity_concurrency(
                    traffic,
                    &ReplayRoleConfig {
                        dp_size: engine_config.dp_size,
                        tensor_parallel_size: engine_config.tensor_parallel_size,
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
                validate_public_agentic_engine(&built.input, &engine_config.rank)?;
            }
            let resolved_basis = built_input
                .as_ref()
                .and_then(|built| built.weka_nested_timestamp_basis);
            let input = built_input.map(|built| built.input);
            let factory = timing.map_or_else(
                ReplayEngineFactory::new,
                ReplayEngineFactory::with_timing_model,
            );
            run_with_input(spec, factory, input, capture_artifacts)
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
            )?;
            let decode_timing = resolve_role_timing(
                &mut decode,
                decode_capacity_is_explicit,
                ForwardPassWorkerType::Decode,
                capture_performance_diagnostics,
            )?;
            if let Some(timing) = prefill_timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &prefill.rank));
            }
            if let Some(timing) = decode_timing.as_ref() {
                power_sources.push(TimingPowerSource::new(Arc::clone(timing), &decode.rank));
            }
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
                for role in [&engine_config.prefill, &engine_config.decode] {
                    validate_public_agentic_engine(
                        &built.input,
                        &role.as_ref().expect("P/D role was materialized").rank,
                    )?;
                }
            }
            let resolved_basis = built_input
                .as_ref()
                .and_then(|built| built.weka_nested_timestamp_basis);
            let input = built_input.map(|built| built.input);
            run_with_input(
                spec,
                ReplayEngineFactory::with_optional_role_timing_models(
                    prefill_timing,
                    decode_timing,
                ),
                input,
                capture_artifacts,
            )
            .map(|(report, artifacts)| (report, artifacts, resolved_basis))
        }
    }
    .context("AISimulate replay failed")?;
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
    if capture_performance_diagnostics {
        report_json["performance_diagnostics"] =
            replay_performance_diagnostics(timing_evidence.as_ref());
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
        let object = report_json
            .as_object_mut()
            .context("AISimulate replay report did not serialize as an object")?;
        object.insert(
            "per_request".to_string(),
            serde_json::to_value(&report.per_request)
                .context("serializing AISimulate per-request report records")?,
        );
    }
    let output = if let Some(artifacts) = artifacts {
        serde_json::json!({
            "report": report_json,
            "artifacts": artifacts,
        })
    } else {
        report_json
    };
    serde_json::to_string(&output).context("serializing AISimulate replay output")
}

fn replay_python_error(error: anyhow::Error) -> PyErr {
    if error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<crate::replay::ReplayError>(),
            Some(crate::replay::ReplayError::ResourceLimited(_))
        )
    }) {
        PyMemoryError::new_err(format!("{error:#}"))
    } else {
        PyRuntimeError::new_err(format!("{error:#}"))
    }
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

/// AISimulate native runtime module.
#[pymodule]
fn _runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
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
            for (options, message) in [
                (serde_json::json!({"backend": "trtllm"}), "vLLM and SGLang"),
                (
                    serde_json::json!({"aic_nextn": 1}),
                    "speculative decoding disabled",
                ),
                (
                    serde_json::json!({
                        "kv_cache_bytes_per_token": 16,
                        "native_host_offload": {"num_host_blocks": 8}
                    }),
                    "HBM-only",
                ),
            ] {
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
            decoder_replay: false,
            phase_provider: AicPhaseProvider::Python,
            use_fpm_decode_totals,
            fpm_decode_kv_ceiling: None,
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            phase_cache: quick_cache::sync::Cache::new(128),
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
        config.backend_version = Some("0.5.17".into());
        config.tp = 16;
        config.moe_tp_size = Some(4);
        config.moe_ep_size = Some(4);
        config.systems_paths = vec![
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../python/aisimulate/src/aisimulate_core/systems"),
        ];
        let mut native =
            AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated).unwrap();
        native.diagnostic_model = None;
        let raw = match &native.phase_provider {
            AicPhaseProvider::Native(engine) => Arc::clone(engine),
            AicPhaseProvider::Python | AicPhaseProvider::NativeEntries(_) => unreachable!(),
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
                        let a = actual.as_phase();
                        let b = expected.as_phase();
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
        let mut engine = serde_json::json!({"rank": {"timing_model": {"type":"external", "provider":"aic", "config":{"tp":2, "attention_dp":4}}}});
        infer_aic_timing_topology(&mut engine).unwrap();
        assert_eq!(engine["tensor_parallel_size"], 2);
        assert_eq!(engine["dp_size"], 4);
        engine["tensor_parallel_size"] = serde_json::json!(1);
        assert!(infer_aic_timing_topology(&mut engine).is_err());
    }

    #[test]
    fn estimator_request_carries_both_context_parallel_knobs() {
        // The native estimator path builds the engine from this request, so a
        // dropped knob would silently price a dcp=1 engine for a dcp=8 worker.
        let mut config = aic_config();
        config.cp_size = Some(2);
        config.dcp_size = Some(4);
        let request = config
            .estimator_request(ForwardPassWorkerType::Aggregated)
            .unwrap();
        assert_eq!(request.cp_size, Some(2));
        assert_eq!(request.dcp_size, Some(4));
        let unset = aic_config()
            .estimator_request(ForwardPassWorkerType::Aggregated)
            .unwrap();
        assert_eq!((unset.cp_size, unset.dcp_size), (None, None));
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

    fn aic_config() -> AicTimingConfig {
        AicTimingConfig {
            model: "test-model".into(),
            backend: "vllm".into(),
            system: "test-system".into(),
            tp: 1,
            backend_version: None,
            pp: 1,
            attention_dp: 1,
            moe_tp_size: None,
            moe_ep_size: None,
            cp_size: None,
            dcp_size: None,
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
            estimator_config: EstimatorConfig::default(),
            database_mode: crate::DatabaseMode::default(),
            transfer_policy: None,
            systems_paths: Vec::new(),
            attention_backend: None,
            moe_backend: None,
            enable_eplb: false,
            wideep_num_slots: None,
            enable_shared_layer: None,
            strict_provenance: false,
            systems_path: None,
            forward_model: None,
            fpm_parquet_path: None,
            decoder_replay: false,
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
        config.dcp_size = Some(8);
        config.validate_parallel_shape().unwrap();
        // ... decode CP does not: without prefill CP the widths no longer match.
        config.cp_size = None;
        assert!(config.validate_parallel_shape().is_err());
        // Zero is rejected like every other parallel size.
        config.cp_size = Some(8);
        config.dcp_size = Some(0);
        assert!(config.validate_parallel_shape().is_err());
    }

    #[test]
    fn ngram_timing_requires_matching_scheduler_depth_and_no_mtp() {
        let mut config = aic_config();
        config.speculation = Some(crate::ForwardPassSpeculationConfig::Ngram {
            num_speculative_tokens: 2,
        });
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        assert!(materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).is_err());
        role.rank.aic_nextn = Some(2);
        materialize_aic_capacity(&config, &mut role, true, |_, _| unreachable!()).unwrap();
        config.nextn = 2;
        assert!(config.validate_parallel_shape().is_err());
        config.nextn = 0;
        config.speculation = Some(crate::ForwardPassSpeculationConfig::Ngram {
            num_speculative_tokens: 6,
        });
        assert!(config.validate_parallel_shape().is_err());
    }

    #[test]
    fn ngram_timing_rejects_unmodeled_trigger_rate_and_unknown_schemes() {
        for payload in [
            serde_json::json!({"kind": "mtp", "params": {"num_speculative_tokens": 2}}),
            serde_json::json!({"kind": "ngram", "params": {"num_speculative_tokens": 2, "trigger_rate": 0.5}}),
        ] {
            assert!(
                serde_json::from_value::<crate::ForwardPassSpeculationConfig>(payload).is_err()
            );
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
        assert_eq!(summary.as_phase().energy_wms, Some(50_000.0));
        assert_eq!(summary.as_phase().latency_ms, 125.0);
        assert_eq!(summary.as_phase().covered_latency_ms, 100.0);
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
            let err = AicTimingModel::build(&mut config, ForwardPassWorkerType::Aggregated)
                .err()
                .expect("invalid path");
            assert!(err.to_string().contains("fpm_parquet_path"), "{err}");
        }
    }

    #[test]
    fn aic_timing_config_accepts_fpm_forward_model() {
        let config = serde_json::from_value::<AicTimingConfig>(serde_json::json!({
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 1,
            "forward_model": "fpm",
            "fpm_parquet_path": "/artifacts/reviewed-fpm.parquet"
        }))
        .unwrap();
        assert_eq!(config.forward_model.as_deref(), Some("fpm"));
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
            assert!(timing.prefill_batch_validation_can_fail());
            let error = timing.validate_prefill_batch(&geometry).unwrap_err();
            assert!(error.to_string().contains("heterogeneous prefill"));
            timing.decoder_replay = false;
            assert!(!timing.prefill_batch_validation_can_fail());
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
        assert_eq!(cached.as_phase().energy_wms, Some(1_600.0));
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
        timing.diagnostic_model = Some(ForwardPassPerfModel::from_engine(
            native,
            ForwardPassPerfOptions::default(),
        ));
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
                        .as_phase()
                        .latency_ms,
                    0.0
                );
            }
            assert_eq!(
                timing
                    .predict_phase_evidence(1, 128, 2, 128, "static_ctx")
                    .unwrap()
                    .as_phase()
                    .latency_ms,
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
                (Ok(native), Ok(checked)) => assert_bits(native.as_phase(), checked.as_phase()),
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
