// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Includes changes adapted from:
// https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/tests/public-api/src/lib.rs

//! Compile-time contract tests from an external crate's point of view.

use std::path::Path;

use aiconfigurator_core::{
    AicEngine, AicEngineBuilder, AicError, BackendKind, DatabaseMode, EngineConfig,
    ForwardPassPerfModel, ForwardPassPerfOptions, ForwardPassRegressionStoreDiagnostics,
    ForwardPassWorkerType, KvCacheEstimateRequest,
};

/// Compile the ergonomic engine builder without starting embedded Python.
pub fn configured_builder() -> AicEngineBuilder {
    AicEngineBuilder::new("Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm)
        .backend_version("0.10.2")
        .tp_size(2)
        .pp_size(1)
        .attention_dp_size(1)
        .moe_parallelism(None, None)
        .gemm_quant_mode("bfloat16")
        .moe_quant_mode("bfloat16")
        .kvcache_quant_mode("bfloat16")
        .fmha_quant_mode("bfloat16")
        .comm_quant_mode("bfloat16")
        .database_mode(DatabaseMode::Empirical)
        .shared_layer(true)
        .transfer_policy(vec!["xshape".to_owned()])
        .strict_provenance(true)
        .speculative_decoding(0)
        .kv_block_size(16)
        .systems_path("/tmp/systems")
}

/// Compile the builder's terminal operation as an external consumer would.
/// The function is intentionally not called by the tests because it embeds
/// Python and needs installed model/system data.
pub fn build_engine(builder: AicEngineBuilder) -> Result<AicEngine, AicError> {
    builder.build()
}

/// Compile the forward-pass model's public constructor and telemetry type.
pub fn regression_model() -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::from_regression(ForwardPassWorkerType::Aggregated, regression_options())
}

/// Per-store diagnostics remain accessible without changing the summary type.
pub fn regression_stores(
    model: &ForwardPassPerfModel,
) -> Vec<ForwardPassRegressionStoreDiagnostics> {
    model.regression_store_diagnostics()
}

/// Construct and expose every public regression-weight option from an external crate.
pub fn regression_options() -> ForwardPassPerfOptions {
    ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_prefill_attention_pair_weight: 3.0,
        regression_ffn_token_weight: 4.0,
        ..ForwardPassPerfOptions::default()
    }
}

/// Compile the fallback-capable constructor with its required worker type.
/// This is not called because native construction embeds Python.
pub fn best_available_model(config: EngineConfig) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available(
        config,
        ForwardPassWorkerType::Decode,
        ForwardPassPerfOptions::default(),
    )
}

/// Compile the explicit-systems-root constructor from an external crate.
pub fn best_available_model_with_roots(
    config: EngineConfig,
    systems_root: impl AsRef<Path>,
) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available_with_roots(
        config,
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions::default(),
        systems_root,
    )
}

/// Keep the KV request type in the external-consumer contract without
/// constructing an environment-dependent estimate.
pub fn accept_kv_request(request: KvCacheEstimateRequest) -> KvCacheEstimateRequest {
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiconfigurator_core::{
        ForwardPassMetrics, ForwardPassRegressionWorkloadKind, TimingEvidenceSource,
        TimingEvidenceSummary, TimingOperationEvidence, TimingPhaseEvidence,
        ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION, FPM_VERSION,
    };

    #[test]
    fn schema_constants_and_metric_defaults_are_public() {
        assert_eq!(ENGINE_CONFIG_SCHEMA_VERSION, 1);
        // v5: MlaModuleOp gained native_num_heads (#1458).
        // v6: Kda op variant appended (Kimi-K3; renumbered at the merge).
        // v7: MoEDispatchOp gained attn_ar_modeled.
        // v8: GemmOp gained below_grid_sol.
        // v9: FpmForward whole-model variant appended (renumbered at each
        // merge from concurrent claims of v5/v7/v8).
        // v10: MhcModuleOp gained seq_split (issue #1498; renumbered at the
        // rebase from a concurrent claim of v7).
        // v11: wideEP MoE variants removed, MoeAllToAll/MoeExpertCompute
        // appended after FpmForward; MoeExpertComputeOp gained enable_eplb
        // (AIC-1601).
        // v12: DsaModuleOp gained attn_projection_quant_modes (PR-6 weight
        // physics) — a positional bincode op-layout change.
        // v13: the engine owns shared-layer source resolution — EngineConfig
        // dropped the Python-resolved perf_db_sources map (a bincode
        // config-layout change) for enable_shared_layer + strict_provenance
        // (deprecation-cleanup PR).
        // v14: GdnOp gained mamba_ssm_dtype (PR #1533) — a positional
        // bincode op-layout change.
        // v15: Context/GenerationAttentionOp gained lane_order (AIC-1715/1716;
        //     renumbered from its own branch's concurrent v8/v9/v10/v12/v14
        //     claims at merge with #1503/#1461/issue #1498/PR-6/#1533).
        // v16: GenerationAttentionOp gained use_qk_norm (Muse Glimmer
        //     continuation) — a positional bincode op-layout change.
        // v17: ContextAttentionOp gained apply_rope (Muse Glimmer review
        //     follow-up) — a positional bincode op-layout change.
        // v18: speculative attention width fields and FpmForward verify_width.
        assert_eq!(ENGINE_SPEC_SCHEMA_VERSION, 18);
        assert_eq!(FPM_VERSION, 1);
        assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
    }

    struct LatencyOnlyProvider;

    impl aiconfigurator_core::TimingModel for LatencyOnlyProvider {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(1.0)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(2.0)
        }
    }

    #[test]
    fn external_latency_only_provider_needs_no_energy_implementation() {
        use aiconfigurator_core::TimingModel;
        assert_eq!(LatencyOnlyProvider.evidence_summary(), None);
        assert_eq!(
            LatencyOnlyProvider.predict_prefill_ms(1, 128, 0).unwrap(),
            1.0
        );
    }

    #[test]
    fn timing_evidence_types_are_public() {
        let operation =
            TimingOperationEvidence::new("gemm", 2.0, Some(900.0), TimingEvidenceSource::Silicon)
                .unwrap();
        let phase = TimingPhaseEvidence::from_operations(vec![operation]);
        let summary = TimingEvidenceSummary {
            prefill: phase,
            decode: TimingPhaseEvidence::default(),
        };
        assert_eq!(summary.prefill.energy_wms, Some(900.0));
    }

    #[test]
    fn power_statistics_require_validated_public_construction() {
        use aiconfigurator_core::replay::TracePowerStats;

        let available = TracePowerStats::new(Some(500.0), 0.9).unwrap();
        assert_eq!(available.power_w(), Some(500.0));
        assert_eq!(available.coverage(), 0.9);
        let withheld = TracePowerStats::new(None, 0.42).unwrap();
        assert_eq!(withheld.power_w(), None);
        assert_eq!(withheld.coverage(), 0.42);
        assert!(TracePowerStats::new(None, 1.0).is_ok());

        for (watts, coverage) in [
            (Some(500.0), 0.9_f64.next_down()),
            (Some(f64::NAN), 1.0),
            (Some(f64::INFINITY), 1.0),
            (Some(0.0), 1.0),
            (Some(-1.0), 1.0),
            (None, f64::NAN),
            (None, f64::INFINITY),
            (None, -0.1),
            (None, 1.1),
        ] {
            assert!(TracePowerStats::new(watts, coverage).is_err());
        }
    }

    #[test]
    fn ergonomic_builder_is_available_to_external_crates() {
        let _builder = configured_builder();
    }

    #[test]
    fn regression_constructor_is_environment_independent() {
        let model = regression_model().expect("construct regression model");
        let stores = regression_stores(&model);
        assert_eq!(stores.len(), 4);
        assert_eq!(
            stores[0].workload_kind,
            ForwardPassRegressionWorkloadKind::PureDecode
        );
        assert!(stores
            .iter()
            .all(|store| !store.ready && store.retained_observations == 0));
        let _roles = [
            ForwardPassWorkerType::Prefill,
            ForwardPassWorkerType::Decode,
            ForwardPassWorkerType::Aggregated,
        ];
    }

    #[test]
    fn regression_weight_fields_are_public() {
        let options = regression_options();
        assert_eq!(options.regression_attention_kv_weight, 2.0);
        assert_eq!(options.regression_prefill_attention_pair_weight, 3.0);
        assert_eq!(options.regression_ffn_token_weight, 4.0);
    }
}
