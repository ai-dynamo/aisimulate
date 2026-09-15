// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-only PyO3 boundary for one materialized AISimulate replay execution.

use std::path::PathBuf;
use std::sync::Arc;

use crate::engine::{Backend, EngineConfig, TimingModel, TimingModelConfig};
use crate::replay::{
    ReplayArtifactKvEventVisibility, ReplayArtifacts, ReplayEngineConfig, ReplayEngineFactory,
    ReplayRoleConfig, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
    loadgen::{
        AgenticSnapshotOptions, ArrivalSpec, DelaySpec, DynamoRequestTrace, LengthSpec,
        SyntheticTraceSpec, Trace, ValidatedAgenticGraph, WekaImportOptions,
        WekaNestedTimestampBasis, WekaResolvedTimestampBasis, WorkloadDriver,
        load_agentic_mooncake, load_weka_agentic_graph_with_options,
    },
};
use anyhow::{Context, Result, anyhow, ensure};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExecutionPayload {
    Configured {
        spec: ReplaySpec,
        traffic: Box<RuntimeTraffic>,
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
    isl: Option<usize>,
    #[serde(default)]
    osl: Option<usize>,
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
    #[serde(alias = "tp_size")]
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
    #[serde(default, alias = "gemm_quant_mode")]
    gemm_dtype: Option<String>,
    #[serde(default, alias = "moe_quant_mode")]
    moe_dtype: Option<String>,
    #[serde(default, alias = "fmha_quant_mode")]
    fmha_dtype: Option<String>,
    #[serde(default, alias = "kvcache_quant_mode")]
    kv_cache_dtype: Option<String>,
    #[serde(default, alias = "comm_quant_mode")]
    comm_dtype: Option<String>,
    #[serde(default)]
    nextn: u32,
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
}

const fn one() -> u32 {
    1
}

impl AicTimingConfig {
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
                && self.moe_ep_size != Some(0),
            "AIC timing parallel sizes tp, pp, attention_dp, moe_tp_size, and \
             moe_ep_size must be positive"
        );
        ensure!(self.nextn <= 5, "AIC nextn must be in 0..=5");
        ensure!(
            self.moe_tp_size.is_some() == self.moe_ep_size.is_some(),
            "AIC moe_tp_size and moe_ep_size must be configured together"
        );
        if let (Some(moe_tp), Some(moe_ep)) = (self.moe_tp_size, self.moe_ep_size) {
            ensure!(
                u64::from(self.tp) * u64::from(self.attention_dp)
                    == u64::from(moe_tp) * u64::from(moe_ep),
                "AIC topology requires tp * attention_dp == moe_tp_size * moe_ep_size"
            );
        }
        Ok(())
    }
}

struct AicTimingModel {
    engine: Py<PyAny>,
    use_fpm_decode_totals: bool,
    fpm_decode_kv_ceiling: Option<u32>,
}

impl AicTimingModel {
    fn build(config: AicTimingConfig) -> Result<Self> {
        ensure!(
            !config.model.trim().is_empty(),
            "AIC timing config field \"model\" cannot be empty"
        );
        ensure!(
            !config.system.trim().is_empty(),
            "AIC timing config field \"system\" cannot be empty"
        );
        config.validate_parallel_shape()?;
        ensure!(
            matches!(config.backend.as_str(), "vllm" | "sglang" | "trtllm"),
            "unsupported AIC backend {:?}; expected vllm, sglang, or trtllm",
            config.backend
        );
        ensure!(
            !config.resolved_backend_version().is_empty(),
            "AIC backend version cannot be empty"
        );
        config.resolved_memory_fraction()?;

        let use_fpm_decode_totals = config.forward_model.as_deref() == Some("fpm");
        let (engine, fpm_decode_kv_ceiling) = Python::with_gil(|py| -> PyResult<_> {
            let sdk = PyModule::import(py, "aiconfigurator_core.sdk.engine")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend_version", config.resolved_backend_version())?;
            kwargs.set_item("tp_size", config.tp)?;
            kwargs.set_item("pp_size", config.pp)?;
            kwargs.set_item("attention_dp_size", config.attention_dp)?;
            kwargs.set_item("moe_tp_size", config.moe_tp_size)?;
            kwargs.set_item("moe_ep_size", config.moe_ep_size)?;
            kwargs.set_item("gemm_quant_mode", config.gemm_dtype.as_deref())?;
            kwargs.set_item("moe_quant_mode", config.moe_dtype.as_deref())?;
            kwargs.set_item("fmha_quant_mode", config.fmha_dtype.as_deref())?;
            kwargs.set_item("kvcache_quant_mode", config.kv_cache_dtype.as_deref())?;
            kwargs.set_item("comm_quant_mode", config.comm_dtype.as_deref())?;
            kwargs.set_item("nextn", config.nextn)?;
            kwargs.set_item("kv_block_size", config.kv_block_size)?;
            kwargs.set_item("systems_path", config.systems_path.as_deref())?;
            kwargs.set_item("forward_model", config.forward_model.as_deref())?;
            let spec = sdk.getattr("compile_engine")?.call(
                (
                    config.model.as_str(),
                    config.system.as_str(),
                    config.backend.as_str(),
                ),
                Some(&kwargs),
            )?;
            let aic = PyModule::import(py, "aiconfigurator_core")?
                .getattr("AicEngine")?
                .call_method1("from_spec", (spec, config.systems_path.as_deref()))?;
            let fpm_decode_kv_ceiling = if use_fpm_decode_totals {
                aic.call_method0("fpm_decode_kv_ceiling")?
                    .extract::<Option<u32>>()?
            } else {
                None
            };
            Ok((aic.unbind(), fpm_decode_kv_ceiling))
        })
        .map_err(|error| {
            anyhow!("AIC timing provider could not compile the requested engine: {error}")
        })?;
        Ok(Self {
            engine,
            use_fpm_decode_totals,
            fpm_decode_kv_ceiling,
        })
    }
}

impl TimingModel for AicTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        let batch_size = checked_u32(batch_size, "prefill batch size")?;
        let mean_isl = checked_u32(mean_isl, "mean input length")?;
        let mean_prefix = checked_u32(mean_prefix, "mean prefix length")?;
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
        let mean_context_length = checked_u32(mean_context_length, "mean context length")?;
        Python::with_gil(|py| {
            self.engine
                .bind(py)
                .call_method1(
                    "predict_decode_latency",
                    (batch_size, mean_context_length, 2),
                )?
                .extract::<f64>()
        })
        .map_err(|error| anyhow!("AIC decode prediction failed: {error}"))
    }
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} {value} exceeds AIC's u32 limit"))
}

fn estimate_aic_num_gpu_blocks(config: &AicTimingConfig, role: &ReplayRoleConfig) -> Result<usize> {
    let (memory_fraction_kind, memory_fraction_value) = config.resolved_memory_fraction()?;
    Python::with_gil(|py| -> PyResult<usize> {
        let memory = PyModule::import(py, "aiconfigurator_core.sdk.memory")?;
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
        kwargs.set_item("gemm_quant_mode", config.gemm_dtype.as_deref())?;
        kwargs.set_item("moe_quant_mode", config.moe_dtype.as_deref())?;
        kwargs.set_item("fmha_quant_mode", config.fmha_dtype.as_deref())?;
        kwargs.set_item("kvcache_quant_mode", config.kv_cache_dtype.as_deref())?;
        kwargs.set_item("comm_quant_mode", config.comm_dtype.as_deref())?;
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
    ensure!(
        config.nextn as usize == engine_nextn,
        "AIC nextn={} does not match engine aic_nextn={engine_nextn}",
        config.nextn
    );
    if capacity_is_explicit {
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
    if capacity_is_explicit {
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
        .is_some_and(|rank| rank.contains_key("num_gpu_blocks"))
}

fn resolve_role_timing(
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
) -> Result<Option<Arc<dyn TimingModel>>> {
    let TimingModelConfig::External { provider, config } = role.rank.timing_model.clone() else {
        return Ok(None);
    };
    ensure!(
        provider == "aic",
        "native timing provider {provider:?} is not installed; only \"aic\" is \
         available in the AISimulate runtime"
    );
    let config: AicTimingConfig =
        serde_json::from_value(config).context("invalid AIC timing provider configuration")?;
    materialize_aic_capacity(
        &config,
        role,
        capacity_is_explicit,
        estimate_aic_num_gpu_blocks,
    )?;
    let timing = AicTimingModel::build(config)?;
    cap_role_capacity_to_fpm_decode_domain(
        role,
        timing.fpm_decode_kv_ceiling,
        capacity_is_explicit,
    )?;
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
        WorkloadDriver::new_agentic_snapshots(prepared, engine_block_size, true, speedup)
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
    allow_agentic: bool,
) -> Result<BuiltRuntimeInput> {
    ensure!(engine_block_size > 0, "engine block size must be positive");
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
            ensure!(allow_agentic, "agentic trace requires aggregated topology");
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
                allow_agentic,
                "Weka agentic trace requires aggregated topology"
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
                        allow_agentic,
                        "agentic Dynamo trace requires aggregated topology"
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
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: engine_block_size,
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

fn execute_json(payload: &str, capture_artifacts: bool) -> Result<String> {
    let (mut spec, mut traffic) =
        match serde_json::from_str(payload).context("invalid AISimulate execution ReplaySpec")? {
            ExecutionPayload::Configured { spec, traffic } => (spec, Some(*traffic)),
            ExecutionPayload::Legacy(spec) => (spec, None),
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
    let mut engine_config: ReplayEngineConfig = if spec.engine.is_null() {
        ReplayEngineConfig::default()
    } else {
        serde_json::from_value(spec.engine.clone())
            .context("invalid native engine descriptor in execution ReplaySpec")?
    };

    let (report, artifacts, resolved_weka_timestamp_basis) = match spec.topology.clone() {
        ReplayTopology::Aggregated { .. } => {
            let mut role = aggregated_role(&engine_config);
            let capacity_is_explicit = role
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, None));
            let timing = resolve_role_timing(&mut role, capacity_is_explicit)?;
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
                .map(|traffic| build_runtime_input(traffic, engine_config.rank.block_size, true))
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
            let prefill_capacity_is_explicit = prefill
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, Some("prefill")));
            let decode_capacity_is_explicit = decode
                .num_gpu_blocks_is_explicit
                .unwrap_or_else(|| role_capacity_is_explicit(&serialized_engine, Some("decode")));
            let prefill_timing = resolve_role_timing(&mut prefill, prefill_capacity_is_explicit)?;
            let decode_timing = resolve_role_timing(&mut decode, decode_capacity_is_explicit)?;
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
            let built_input = traffic
                .map(|traffic| {
                    build_runtime_input(
                        traffic,
                        engine_config
                            .prefill
                            .as_ref()
                            .expect("prefill role was materialized")
                            .rank
                            .block_size,
                        false,
                    )
                })
                .transpose()?;
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
    let mut report_json =
        serde_json::to_value(&report).context("serializing AISimulate replay report summary")?;
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
    if !report.per_request.is_empty() {
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

/// Execute one canonical serialized ReplaySpec and return serialized report JSON.
#[pyfunction]
fn run_replay_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, false))
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))
}

/// Execute one fixed aggregated ReplaySpec and return report plus parity artifacts.
#[pyfunction]
fn run_replay_with_artifacts_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload, true))
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))
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
                build_runtime_input(traffic, 64, true)
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
                } else {
                    let report: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
                    assert_eq!(report["completed_requests"], 1);
                    assert!(report.get("agentic_qualification").is_none());
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
            gemm_dtype: None,
            moe_dtype: None,
            fmha_dtype: None,
            kv_cache_dtype: None,
            comm_dtype: None,
            nextn: 0,
            kv_block_size: None,
            gpu_memory_utilization: None,
            mem_fraction_static: None,
            free_gpu_memory_fraction: None,
            cuda_graph_reserved_bytes: 0,
            systems_path: None,
            forward_model: None,
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
    fn aic_timing_config_accepts_fpm_forward_model() {
        let config = serde_json::from_value::<AicTimingConfig>(serde_json::json!({
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 1,
            "forward_model": "fpm"
        }))
        .unwrap();
        assert_eq!(config.forward_model.as_deref(), Some("fpm"));
    }

    #[test]
    fn fpm_decode_timing_queries_exact_past_kv_total() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = AicTimingModel {
            engine,
            use_fpm_decode_totals: true,
            fpm_decode_kv_ceiling: None,
        };

        let latency = timing
            .predict_decode_ms(35, 546_081, 15_602, 546_048)
            .unwrap();

        assert_eq!(latency, 546_046.0);
    }

    #[test]
    fn fpm_decode_timing_caps_logical_past_kv_at_physical_capacity() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = AicTimingModel {
            engine,
            use_fpm_decode_totals: true,
            fpm_decode_kv_ceiling: None,
        };

        let latency = timing
            .predict_decode_ms(35, 546_116, 15_603, 546_048)
            .unwrap();

        assert_eq!(latency, 546_048.0);
    }

    #[test]
    fn op_level_decode_timing_keeps_legacy_mean_coordinate() {
        pyo3::prepare_freethreaded_python();
        let engine = Python::with_gil(|py| Py::new(py, DecodeCoordinateProbe).unwrap().into_any());
        let timing = AicTimingModel {
            engine,
            use_fpm_decode_totals: false,
            fpm_decode_kv_ceiling: None,
        };

        let latency = timing
            .predict_decode_ms(35, 546_081, 15_602, 546_048)
            .unwrap();

        assert_eq!(latency, 546_105.0);
    }

    #[test]
    fn json_bindings_share_report_and_retain_request_correlation() {
        let spec = ReplaySpec {
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
    }
}
