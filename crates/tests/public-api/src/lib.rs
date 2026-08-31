// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time contract tests from an external crate's point of view.

use aiconfigurator_core::{
    AicEngine, AicEngineBuilder, AicError, BackendKind, ForwardPassFallbackPolicy,
    ForwardPassModelKind, ForwardPassPerfModel, ForwardPassPerfModelConfig,
    ForwardPassPerfOptions, KvCacheEstimateRequest,
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

/// Compile the forward-pass model's sole public production constructor.
///
/// The function is intentionally not called because construction loads model
/// and systems data. Its signature proves an external crate can use the
/// canonical config/options boundary without reaching internal constructors.
pub fn forward_pass_model(
    config: ForwardPassPerfModelConfig,
    options: Option<ForwardPassPerfOptions>,
) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available(config, options)
}

/// Compile construction of the canonical config from an external crate.
pub fn forward_pass_config() -> ForwardPassPerfModelConfig {
    ForwardPassPerfModelConfig {
        model: "Qwen/Qwen3-32B".into(),
        system: "h200_sxm".into(),
        backend: BackendKind::Vllm,
        backend_version: Some("0.10.2".into()),
        tp: 2,
        pp: 1,
        attention_dp: 1,
        moe_tp_size: None,
        moe_ep_size: None,
        gemm_quant_mode: None,
        moe_quant_mode: None,
        fmha_quant_mode: None,
        kvcache_quant_mode: None,
        comm_quant_mode: None,
        nextn: 0,
        kv_block_size: Some(16),
        forward_model: ForwardPassModelKind::OpLevel,
        database_mode: Default::default(),
        transfer_policy: None,
        systems_paths: Vec::new(),
        fallback_policy: ForwardPassFallbackPolicy::Error,
    }
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
        ForwardPassMetrics, ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION, FPM_VERSION,
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
        assert_eq!(ENGINE_SPEC_SCHEMA_VERSION, 13);
        assert_eq!(FPM_VERSION, 1);
        assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
    }

    #[test]
    fn ergonomic_builder_is_available_to_external_crates() {
        let _builder = configured_builder();
    }

    #[test]
    fn canonical_forward_pass_config_is_external() {
        let config = forward_pass_config();
        assert_eq!(config.tp, 2);
        let _constructor = forward_pass_model;
    }
}
