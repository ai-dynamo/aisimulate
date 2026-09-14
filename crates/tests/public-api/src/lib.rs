// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Includes changes adapted from:
// https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/tests/public-api/src/lib.rs

//! Compile-time contract tests from an external crate's point of view.

use std::path::Path;
use std::sync::Arc;

use aiconfigurator_core::replay::loadgen::{AgenticPromptMaterializer, ValidatedAgenticGraph};
use aiconfigurator_core::{
    AicEngine, AicEngineBuilder, AicError, BackendKind, DatabaseMode, EngineConfig,
    ForwardPassPerfModel, ForwardPassPerfOptions, ForwardPassWorkerType, KvCacheEstimateRequest,
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

/// Retain the validated graph's shared prompt identity from an external crate.
pub fn agentic_prompt_materializer(
    graph: &ValidatedAgenticGraph,
) -> &Arc<AgenticPromptMaterializer> {
    graph.prompt_materializer()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiconfigurator_core::replay::loadgen::{
        AgenticGraphBuilder, AgenticHashIdScope, AgenticMooncakeHeader, AgenticMooncakeRow,
        AgenticSourceProvenance, ReplayRequestHashes, AGENTIC_MOONCAKE_SCHEMA,
        AGENTIC_MOONCAKE_VERSION,
    };
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

    #[test]
    fn ergonomic_builder_is_available_to_external_crates() {
        let _builder = configured_builder();
    }

    #[test]
    fn regression_constructor_is_environment_independent() {
        let _model = regression_model().expect("construct regression model");
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

    #[test]
    fn agentic_prefix_materialization_is_available_to_external_crates() {
        let mut builder = AgenticGraphBuilder::new(AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 64,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "public-api-fixture".to_string(),
                digest: "self-authored-prefix-fixture".to_string(),
            },
        })
        .expect("valid graph header");
        builder
            .push(AgenticMooncakeRow {
                request_id: "request".to_string(),
                play_id: "play".to_string(),
                session_id: "session".to_string(),
                model: "fixture-model".to_string(),
                input_length: Some(129),
                output_length: Some(1),
                hash_ids: Some(vec![90, 10, 50]),
                ..AgenticMooncakeRow::default()
            })
            .expect("valid request row");
        let graph = builder.finish().expect("valid graph");
        let cloned_graph = graph.clone();
        let materializer = agentic_prompt_materializer(&graph);
        assert!(Arc::ptr_eq(
            materializer,
            agentic_prompt_materializer(&cloned_graph)
        ));
        assert_eq!(materializer.block_size(), 64);

        let node = &graph.nodes()[0];
        let full_prompt = materializer
            .materialize_prefix(node, node.input_length())
            .expect("materialize full prompt");
        let prefix = materializer
            .materialize_prefix(node, 65)
            .expect("materialize prefix across source unit boundary");
        assert_eq!(prefix, full_prompt[..65]);
        let hashes: ReplayRequestHashes = materializer
            .replay_hashes(node, 65, 32)
            .expect("re-block prefix for engine");
        assert_eq!(hashes.local_block_hashes.len(), 2);
        assert_eq!(hashes.sequence_hashes.len(), 2);
        assert_eq!(hashes, ReplayRequestHashes::from_tokens(&prefix, 32));
        assert!(materializer
            .materialize_prefix(node, node.input_length() + 1)
            .is_err());
        assert!(materializer.replay_hashes(node, 65, 0).is_err());
    }
}
