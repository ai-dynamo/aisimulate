// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical performance-model timing for runtime-neutral Engine consumers.
//!
//! Construction follows `ForwardPassPerfModel::best_available`; op-level model
//! compilation currently needs the explicit Python feature, while every pass
//! query below runs in Rust. Missing calibration never falls back to constants.

use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};

use super::{IterationMs, MixedPassMs, PrefillTotals, RankWork, TimingModel, VisionShape};
use crate::{
    ForwardPassMetrics, ForwardPassPerfDiagnostics, ForwardPassPerfModel,
    ForwardPassPerfModelConfig, ForwardPassPerfProvenance, ScheduledRequestMetrics,
};

/// A pinned canonical estimator over actual scheduled rank work.
///
/// This bridge intentionally publishes no energy/coverage summary: calibrated
/// latency is not evidence that all requested operation evidence was captured.
pub struct CanonicalTimingModel {
    model: ForwardPassPerfModel,
    config: ForwardPassPerfModelConfig,
}

impl CanonicalTimingModel {
    /// Resolve and pin the canonical estimator. Unsupported speculation, vision,
    /// decoder replay and pipeline layouts fail explicitly rather than being
    /// silently priced with an ordinary text-only aggregate.
    pub fn new(config: ForwardPassPerfModelConfig) -> Result<Arc<Self>> {
        validate_supported_identity(&config)?;
        #[cfg(feature = "python")]
        crate::python::register_embedded_runtime()?;
        let model = ForwardPassPerfModel::best_available(config)?;
        let config = model
            .provenance()
            .ok_or_else(|| anyhow!("canonical model omitted resolved provenance"))?
            .config
            .clone();
        Ok(Arc::new(Self { model, config }))
    }

    /// Construct canonical timing and obtain the existing backend-native memory
    /// model's geometry. No guessed cache reservation or HF download is allowed.
    /// The memory request must name the same hardware/model/backend/precision/
    /// topology and the same worker role.
    #[cfg(feature = "python")]
    pub fn new_with_geometry(
        config: ForwardPassPerfModelConfig,
        memory: crate::KvCacheEstimateRequest,
    ) -> Result<(Arc<Self>, crate::KvCacheEstimate)> {
        validate_supported_identity(&config)?;
        validate_memory_topology(&memory, config.worker_type)?;
        ensure!(
            memory.engine.model_name == config.model
                && memory.engine.system_name == config.system
                && memory.engine.backend == config.backend
                && memory.engine.backend_version == config.backend_version
                && memory.engine.parallel.tp_size == config.tp
                && memory.engine.parallel.pp_size == config.pp
                && memory.engine.parallel.attention_dp_size.unwrap_or(1) == config.attention_dp
                && memory.engine.parallel.moe_tp_size == config.moe_tp_size
                && memory.engine.parallel.moe_ep_size == config.moe_ep_size,
            "memory request conflicts with canonical timing identity"
        );
        ensure!(
            !memory.options.allow_naive_fallback && !memory.options.allow_hf_config_download,
            "canonical GPU geometry requires native memory estimation without heuristic fallback or downloads"
        );
        validate_memory_precision(&config, &memory)?;
        ensure!(
            config.systems_paths.len() == 1
                && memory.engine.systems_path.as_ref() == config.systems_paths.first(),
            "native memory and timing must share the same explicit systems root"
        );
        let model = Self::new(config)?;
        let geometry = crate::perfmodel::memory::estimate_kv_cache(memory)?;
        ensure!(
            geometry.source == crate::EstimateSource::Native,
            "canonical GPU geometry requires the native backend memory model"
        );
        Ok((model, geometry))
    }

    /// Immutable resolved estimator identity and source selection.
    pub fn provenance(&self) -> Option<&ForwardPassPerfProvenance> {
        self.model.provenance()
    }

    /// Canonical readiness and estimator diagnostics, never synthetic evidence.
    pub fn diagnostics(&self) -> ForwardPassPerfDiagnostics {
        self.model.diagnostics()
    }

    /// Profile-backed cache budget when the canonical identity owns resource facts.
    /// Op-level-only identities without profile resource facts fail explicitly.
    pub fn estimate_cache_budget(
        &self,
        request: &crate::perfmodel::FpmCacheBudgetRequest,
    ) -> Result<crate::perfmodel::FpmCacheBudget> {
        Ok(self.config.estimate_cache_budget(request)?)
    }

    fn query(&self, ranks: &[ForwardPassMetrics]) -> Result<f64> {
        let duration = self.model.estimate_forward_pass_time_ms(ranks)?
            .ok_or_else(|| anyhow!("canonical performance model has no ready calibrated estimate for this scheduled pass"))?;
        ensure!(
            duration.is_finite() && duration >= 0.0,
            "canonical performance model returned invalid duration"
        );
        Ok(duration)
    }

    fn single(
        &self,
        prefill: Option<PrefillTotals>,
        decode: Option<(usize, usize)>,
    ) -> Result<f64> {
        ensure!(
            self.config.attention_dp == 1,
            "phase-only timing cannot represent an attention-DP group"
        );
        let mut scheduled = ScheduledRequestMetrics::default();
        if let Some(prefill) = prefill {
            scheduled.num_prefill_requests = u32::try_from(prefill.batch_size)?;
            scheduled.sum_prefill_tokens = u32::try_from(prefill.new_tokens)?;
            scheduled.sum_prefill_kv_tokens = u32::try_from(prefill.prefix_tokens)?;
        }
        if let Some((batch, kv)) = decode {
            scheduled.num_decode_requests = u32::try_from(batch)?;
            scheduled.sum_decode_kv_tokens = u32::try_from(kv)?;
        }
        self.query(&[ForwardPassMetrics {
            scheduled_requests: scheduled,
            ..Default::default()
        }])
    }
}

fn validate_supported_identity(config: &ForwardPassPerfModelConfig) -> Result<()> {
    // Prefill and Decode bind their immutable worker role into the canonical
    // estimator; Aggregated remains the colocated execution role.
    ensure!(config.pp == 1, "canonical Engine timing supports pp=1 only");
    ensure!(
        config.cp_size.unwrap_or(1) == 1 && config.dcp.unwrap_or(1) == 1,
        "canonical Engine timing bridge does not support CP or DCP"
    );
    validate_moe_topology(
        config.tp,
        config.attention_dp,
        config.moe_tp_size,
        config.moe_ep_size,
    )?;
    ensure!(
        config.moe_backend.is_none()
            && config.moe_kernel_source.is_none()
            && !config.enable_eplb
            && config.wideep_num_slots.is_none(),
        "canonical Engine timing bridge does not support explicit MoE kernel controls"
    );
    ensure!(
        config.nextn == 0 && config.speculation.is_none(),
        "canonical Engine timing bridge does not support speculative decoding"
    );
    ensure!(
        config.encoder_parallel.is_none(),
        "canonical Engine timing bridge does not support vision; use the vision-aware provider"
    );
    ensure!(
        !config.decoder_replay,
        "canonical Engine timing bridge does not support decoder replay"
    );
    Ok(())
}

// Explicit MoE dimensions may describe the same tensor-parallel ranks, but
// this bridge does not introduce expert-parallel or cross-attention-DP work.
// The canonical owner retains precision and compiles/prices its MoE operators.
fn validate_moe_topology(
    tp: u32,
    attention_dp: u32,
    moe_tp: Option<u32>,
    moe_ep: Option<u32>,
) -> Result<()> {
    ensure!(
        matches!((moe_tp, moe_ep), (None, None))
            || (tp > 0 && attention_dp == 1 && moe_tp == Some(tp) && moe_ep == Some(1)),
        "canonical Engine timing supports explicit MoE topology only with moe_tp_size=tp, moe_ep_size=1 and attention_dp=1"
    );
    Ok(())
}

#[cfg(any(test, feature = "python"))]
fn validate_memory_precision(
    config: &ForwardPassPerfModelConfig,
    memory: &crate::KvCacheEstimateRequest,
) -> Result<()> {
    let quant = &memory.engine.quantization;
    for (name, timing, memory_dtype) in [
        (
            "weights",
            config.gemm_quant_mode.as_ref(),
            quant.weight_dtype.as_ref(),
        ),
        (
            "attention",
            config.fmha_quant_mode.as_ref(),
            quant.activation_dtype.as_ref(),
        ),
        (
            "KV cache",
            config.kvcache_quant_mode.as_ref(),
            quant.kv_cache_dtype.as_ref(),
        ),
        (
            "MoE",
            config.moe_quant_mode.as_ref(),
            quant.moe_dtype.as_ref(),
        ),
    ] {
        // Both absent preserves the canonical owner's MoE default. An explicit
        // choice on either side must never silently inherit the other's default.
        if name == "MoE" && timing.is_none() && memory_dtype.is_none() {
            continue;
        }
        let timing = timing
            .ok_or_else(|| anyhow!("canonical GPU geometry requires explicit {name} precision"))?;
        let memory_dtype = memory_dtype
            .ok_or_else(|| anyhow!("native memory request requires explicit {name} precision"))?;
        ensure!(
            serde_json::to_value(memory_dtype)?.as_str() == Some(timing.as_str()),
            "memory precision conflicts with canonical timing identity for {name}"
        );
    }
    Ok(())
}

#[cfg(any(test, feature = "python"))]
fn worker_role_name(role: crate::ForwardPassWorkerType) -> &'static str {
    match role {
        crate::ForwardPassWorkerType::Prefill => "prefill",
        crate::ForwardPassWorkerType::Decode => "decode",
        crate::ForwardPassWorkerType::Aggregated => "aggregated",
    }
}

#[cfg(any(test, feature = "python"))]
fn validate_memory_topology(
    memory: &crate::KvCacheEstimateRequest,
    timing_role: crate::ForwardPassWorkerType,
) -> Result<()> {
    // `estimate_kwargs` forwards `engine.extra["worker_type"]` to the native
    // memory estimator, defaulting to `"aggregated"` when absent. Validate that
    // effective role so the estimator's per-role resources match the timing
    // identity, and require Prefill/Decode requests to name their role rather
    // than silently inheriting the aggregate default.
    let memory_role = memory
        .engine
        .extra
        .get("worker_type")
        .map(String::as_str)
        .unwrap_or("aggregated");
    ensure!(
        matches!(memory_role, "prefill" | "decode" | "aggregated"),
        "canonical native memory geometry received unknown worker role {memory_role:?}"
    );
    let timing_role = worker_role_name(timing_role);
    ensure!(
        memory_role == timing_role,
        "canonical native memory geometry worker role {memory_role:?} conflicts with canonical timing role {timing_role:?}"
    );
    let parallel = &memory.engine.parallel;
    ensure!(
        parallel.cp_size.unwrap_or(1) == 1 && parallel.dcp_size.unwrap_or(1) == 1,
        "canonical native memory geometry does not support CP or DCP"
    );
    validate_moe_topology(
        parallel.tp_size,
        parallel.attention_dp_size.unwrap_or(1),
        parallel.moe_tp_size,
        parallel.moe_ep_size,
    )?;
    ensure!(
        memory
            .engine
            .speculative
            .as_ref()
            .is_none_or(|spec| spec.nextn.unwrap_or(0) == 0)
            && !memory.engine.decoder_replay,
        "canonical native memory geometry does not support speculation or decoder replay"
    );
    Ok(())
}

impl TimingModel for CanonicalTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        let new = mean_isl
            .checked_sub(mean_prefix)
            .ok_or_else(|| anyhow!("prefill prefix exceeds input"))?;
        self.predict_prefill_totals_ms(
            batch_size,
            batch_size
                .checked_mul(new)
                .ok_or_else(|| anyhow!("prefill tokens overflow"))?,
            batch_size
                .checked_mul(mean_prefix)
                .ok_or_else(|| anyhow!("prefill prefix overflow"))?,
        )
    }

    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> Result<f64> {
        self.single(
            Some(PrefillTotals {
                batch_size,
                new_tokens,
                prefix_tokens,
            }),
            None,
        )
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        let kv = if self.config.estimation_mode == crate::EstimationMode::FpmInterpolation {
            active_kv_tokens
                .checked_sub(batch_size)
                .ok_or_else(|| anyhow!("decode context excludes current input"))?
                .min(total_kv_tokens)
        } else {
            batch_size
                .checked_mul(mean_context_length)
                .ok_or_else(|| anyhow!("decode KV overflow"))?
        };
        self.single(None, Some((batch_size, kv)))
    }

    fn predict_mixed_pass_ms(
        &self,
        prefill: PrefillTotals,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<Option<MixedPassMs>> {
        let prefill_ms = self.predict_prefill_totals_ms(
            prefill.batch_size,
            prefill.new_tokens,
            prefill.prefix_tokens,
        )?;
        let kv = if self.config.estimation_mode == crate::EstimationMode::FpmInterpolation {
            active_kv_tokens
                .checked_sub(batch_size)
                .ok_or_else(|| anyhow!("decode context excludes current input"))?
                .min(total_kv_tokens)
        } else {
            batch_size
                .checked_mul(mean_context_length)
                .ok_or_else(|| anyhow!("decode KV overflow"))?
        };
        let total = self.single(Some(prefill), Some((batch_size, kv)))?;
        ensure!(
            total >= prefill_ms,
            "canonical mixed timing is less than prefill timing"
        );
        Ok(Some(MixedPassMs {
            prefill_ms,
            decode_ms: total - prefill_ms,
        }))
    }

    fn predict_iteration_ms(&self, ranks: &[RankWork]) -> Result<Option<IterationMs>> {
        ensure!(
            ranks.len() == self.config.attention_dp as usize,
            "scheduled rank count conflicts with canonical attention_dp identity"
        );
        let mut queries = Vec::with_capacity(ranks.len());
        for (position, rank) in ranks.iter().enumerate() {
            ensure!(
                rank.dp_rank as usize == position,
                "scheduled ranks must be complete and ordered"
            );
            let mut scheduled = ScheduledRequestMetrics::default();
            if let Some(prefill) = &rank.prefill {
                scheduled.num_prefill_requests = u32::try_from(prefill.totals.batch_size)?;
                scheduled.sum_prefill_tokens = u32::try_from(prefill.totals.new_tokens)?;
                scheduled.sum_prefill_kv_tokens = u32::try_from(prefill.totals.prefix_tokens)?;
            }
            if let Some(decode) = &rank.decode {
                ensure!(
                    decode.verify_width == 1,
                    "canonical timing bridge does not support speculative scheduled work"
                );
                scheduled.num_decode_requests = u32::try_from(decode.batch_size)?;
                let kv = if self.config.estimation_mode == crate::EstimationMode::FpmInterpolation {
                    decode
                        .active_kv_tokens
                        .checked_sub(decode.batch_size)
                        .ok_or_else(|| anyhow!("decode context excludes current input"))?
                        .min(decode.kv_bound_tokens)
                } else {
                    decode
                        .batch_size
                        .checked_mul(decode.mean_context_length)
                        .ok_or_else(|| anyhow!("decode KV overflow"))?
                };
                scheduled.sum_decode_kv_tokens = u32::try_from(kv)?;
            }
            queries.push(ForwardPassMetrics {
                dp_rank: rank.dp_rank,
                scheduled_requests: scheduled,
                ..Default::default()
            });
        }
        // Preserve the entire per-rank list: the canonical estimator owns DP
        // reduction and pricing. Never average or maximize before querying it.
        Ok(Some(IterationMs::new(self.query(&queries)?)))
    }

    fn predict_vision_ms(&self, _shapes: &[VisionShape]) -> Result<Option<f64>> {
        Err(anyhow!(
            "canonical text timing bridge cannot price vision work"
        ))
    }

    fn reset_evidence(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ForwardPassPerfModelConfig {
        ForwardPassPerfModelConfig::new(
            "model",
            "system",
            crate::BackendKind::Vllm,
            crate::ForwardPassWorkerType::Aggregated,
        )
    }

    #[test]
    fn supported_timing_roles_are_admitted_and_topology_refusals_remain() {
        for role in [
            crate::ForwardPassWorkerType::Prefill,
            crate::ForwardPassWorkerType::Decode,
            crate::ForwardPassWorkerType::Aggregated,
        ] {
            let mut value = config();
            value.worker_type = role;
            assert!(validate_supported_identity(&value).is_ok());
        }
        for change in 0..11 {
            let mut value = config();
            match change {
                0 => value.cp_size = Some(2),
                1 => value.dcp = Some(2),
                2 => value.moe_tp_size = Some(2),
                3 => value.moe_ep_size = Some(2),
                4 => value.moe_backend = Some("default".into()),
                5 => value.moe_kernel_source = Some("default".into()),
                6 => value.enable_eplb = true,
                7 => value.wideep_num_slots = Some(1),
                8 => {
                    value.tp = 8;
                    value.moe_tp_size = Some(4);
                    value.moe_ep_size = Some(2);
                }
                9 => {
                    value.attention_dp = 2;
                    value.moe_tp_size = Some(1);
                    value.moe_ep_size = Some(1);
                }
                _ => value.pp = 2,
            }
            assert!(validate_supported_identity(&value).is_err());
            assert!(CanonicalTimingModel::new(value).is_err());
        }
    }

    #[test]
    fn explicit_moe_precision_and_same_tp_topology_are_admitted() {
        for precision in ["bfloat16", "fp8"] {
            let mut value = config();
            value.moe_quant_mode = Some(precision.into());
            assert!(validate_supported_identity(&value).is_ok());
            value.tp = 8;
            value.moe_tp_size = Some(8);
            value.moe_ep_size = Some(1);
            assert!(validate_supported_identity(&value).is_ok());
        }
        assert!(validate_moe_topology(0, 1, Some(0), Some(1)).is_err());
        assert!(validate_moe_topology(8, 1, Some(4), Some(1)).is_err());
        assert!(validate_moe_topology(8, 1, None, Some(1)).is_err());
    }

    #[test]
    fn memory_moe_precision_must_match_without_discarding_explicit_choices() {
        let memory: crate::KvCacheEstimateRequest = serde_json::from_value(serde_json::json!({
            "engine": {"schema_version":1,"model_name":"model","system_name":"system",
                "backend":"vllm","tp_size":8,"pp_size":1,"moe_tp_size":8,"moe_ep_size":1,
                "weight_dtype":"fp8","activation_dtype":"bfloat16","kv_cache_dtype":"fp8"},
            "max_num_tokens":128,"max_batch_size":2,"kv_cache_memory_fraction":{"OfTotal":0.9},
            "options":{"allow_naive_fallback":false,"allow_hf_config_download":false}
        }))
        .unwrap();
        let mut value = config();
        value.gemm_quant_mode = Some("fp8".into());
        value.fmha_quant_mode = Some("bfloat16".into());
        value.kvcache_quant_mode = Some("fp8".into());
        assert!(validate_memory_topology(&memory, value.worker_type).is_ok());
        assert!(validate_memory_precision(&value, &memory).is_ok());
        for precision in [crate::DataType::Bfloat16, crate::DataType::Fp8] {
            let mut candidate = memory.clone();
            candidate.engine.quantization.moe_dtype = Some(precision.clone());
            assert!(validate_memory_precision(&value, &candidate).is_err());
            value.moe_quant_mode = Some(
                serde_json::to_value(&precision)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into(),
            );
            assert!(validate_memory_precision(&value, &candidate).is_ok());
            assert!(validate_memory_precision(&value, &memory).is_err());
            candidate.engine.quantization.moe_dtype = Some(if precision == crate::DataType::Fp8 {
                crate::DataType::Bfloat16
            } else {
                crate::DataType::Fp8
            });
            assert!(validate_memory_precision(&value, &candidate).is_err());
            value.moe_quant_mode = None;
        }
    }

    #[test]
    fn memory_worker_role_must_match_timing_role_and_topology_refusals_remain() {
        use crate::ForwardPassWorkerType::{Aggregated, Decode, Prefill};

        let base = serde_json::json!({
            "engine": {"schema_version":1,"model_name":"model","system_name":"system",
                "backend":"vllm","backend_version":"0.24.0","tp_size":1,"pp_size":1,
                "weight_dtype":"bfloat16","activation_dtype":"bfloat16","kv_cache_dtype":"bfloat16"},
            "max_num_tokens":128,"max_batch_size":2,"kv_cache_memory_fraction":{"OfTotal":0.9},
            "gpu_memory_capacity_bytes_override":null,"tolerance_fraction":null,
            "options":{"allow_naive_fallback":false,"allow_hf_config_download":false}
        });
        let memory: crate::KvCacheEstimateRequest = serde_json::from_value(base.clone()).unwrap();
        for (timing_role, memory_role) in [
            (Aggregated, None),
            (Aggregated, Some("aggregated")),
            (Prefill, Some("prefill")),
            (Decode, Some("decode")),
        ] {
            let mut candidate = memory.clone();
            if let Some(role) = memory_role {
                candidate
                    .engine
                    .extra
                    .insert("worker_type".into(), role.into());
            }
            assert!(validate_memory_topology(&candidate, timing_role).is_ok());
        }
        for (timing_role, memory_role) in [
            (Aggregated, Some("prefill")),
            (Aggregated, Some("decode")),
            (Prefill, None),
            (Prefill, Some("aggregated")),
            (Prefill, Some("decode")),
            (Decode, None),
            (Decode, Some("prefill")),
            (Decode, Some("aggregated")),
            (Decode, Some("unknown")),
        ] {
            let mut candidate = memory.clone();
            if let Some(role) = memory_role {
                candidate
                    .engine
                    .extra
                    .insert("worker_type".into(), role.into());
            }
            assert!(validate_memory_topology(&candidate, timing_role).is_err());
        }
        for (field, value) in [
            ("cp_size", serde_json::json!(2)),
            ("dcp_size", serde_json::json!(2)),
            ("moe_tp_size", serde_json::json!(2)),
            ("moe_ep_size", serde_json::json!(2)),
        ] {
            let mut wire = base.clone();
            wire["engine"][field] = value;
            let memory = serde_json::from_value(wire).unwrap();
            assert!(validate_memory_topology(&memory, Aggregated).is_err());
        }
    }
}
