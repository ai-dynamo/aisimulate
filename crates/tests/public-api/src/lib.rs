// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Includes changes adapted from:
// https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/rust/tests/public-api/src/lib.rs

//! Compile-time contract tests from an external crate's point of view.

use std::path::Path;

use aisimulate_core::{
    AicEngine, AicEngineBuilder, AicError, BackendKind, DatabaseMode, EstimationMode,
    EstimatorConfig, ForwardPassPerfModel, ForwardPassPerfModelConfig,
    ForwardPassRegressionStoreDiagnostics, ForwardPassWorkerType, KvCacheEstimateRequest,
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
    let mut config = ForwardPassPerfModelConfig::new(
        "test/model",
        "test-system",
        BackendKind::Vllm,
        ForwardPassWorkerType::Aggregated,
    );
    config.estimation_mode = EstimationMode::FpmRegression;
    config.estimator_config = regression_options();
    ForwardPassPerfModel::best_available(config)
}

/// Per-store diagnostics remain accessible without changing the summary type.
pub fn regression_stores(
    model: &ForwardPassPerfModel,
) -> Vec<ForwardPassRegressionStoreDiagnostics> {
    model.regression_store_diagnostics()
}

/// Construct and expose every public regression-weight option from an external crate.
pub fn regression_options() -> EstimatorConfig {
    let mut config = EstimatorConfig::default();
    config.features.attention_kv_weight = 2.0;
    config.features.prefill_attention_pair_weight = 3.0;
    config.features.ffn_token_weight = 4.0;
    config
}

pub fn best_available_model(
    config: ForwardPassPerfModelConfig,
) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available(config)
}

/// The qualified direct-prefill method is on the canonical returned model.
pub fn predict_graph_prefill(
    model: &ForwardPassPerfModel,
    bs: u32,
    isl: u32,
    prefix: u32,
) -> Result<f64, AicError> {
    model.predict_prefill_latency(bs, isl, prefix)
}

pub fn best_available_model_with_roots(
    mut config: ForwardPassPerfModelConfig,
    systems_root: impl AsRef<Path>,
) -> Result<ForwardPassPerfModel, AicError> {
    config.systems_paths = vec![systems_root.as_ref().to_path_buf()];
    ForwardPassPerfModel::best_available(config)
}

/// Keep the KV request type in the external-consumer contract without
/// constructing an environment-dependent estimate.
pub fn accept_kv_request(request: KvCacheEstimateRequest) -> KvCacheEstimateRequest {
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisimulate_core::{
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
        // v19: DeepSeek V4.1 backend layout and appended operation variants.
        // v20: DeepSeek V4.1 FPM selector diagnostics.
        // v21: observed MoE selection and exact prefill graph composites.
        assert_eq!(ENGINE_SPEC_SCHEMA_VERSION, 21);
        assert_eq!(FPM_VERSION, 1);
        assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
    }

    #[test]
    fn direct_graph_api_rejects_an_unselected_regression_model() {
        let model = regression_model().unwrap();
        assert!(predict_graph_prefill(&model, 1, 1024, 0).is_err());
        let mut controls = EstimatorConfig::default();
        controls.op_level.prefill_graph_profile =
            Some("sglang_glm52_nvfp4_vr200_tp4_graph_v1".into());
        controls.op_level.prefill_graph_profile_id =
            Some("829a83e1629ba546dd4bd90e75a2e2496b7fb24ddc8b60dfbf076ba02312cbce".into());
        assert!(controls.op_level.decode_workload_distribution.is_none());
        let mut config = ForwardPassPerfModelConfig::new(
            "missing-model",
            "missing-system",
            BackendKind::Sglang,
            ForwardPassWorkerType::Prefill,
        );
        config.estimator_config = controls;
        // Auto/default correction cannot admit a direct-only graph profile.
        // This fails before Python/model/system lookup despite nonexistent data.
        assert!(ForwardPassPerfModel::best_available(config)
            .err()
            .unwrap()
            .to_string()
            .contains("requires explicit"));
    }

    struct LatencyOnlyProvider;

    impl aisimulate_core::TimingModel for LatencyOnlyProvider {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(1.0)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(2.0)
        }
    }

    #[test]
    fn external_latency_only_provider_needs_no_energy_implementation() {
        use aisimulate_core::TimingModel;
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
        use aisimulate_core::replay::TracePowerStats;

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
        assert_eq!(options.features.attention_kv_weight, 2.0);
        assert_eq!(options.features.prefill_attention_pair_weight, 3.0);
        assert_eq!(options.features.ffn_token_weight, 4.0);
    }

    #[test]
    fn agentic_snapshot_preparation_and_execution_are_public() {
        use aisimulate_core::replay::loadgen::{
            AgenticGraphBuilder, AgenticHashIdScope, AgenticMooncakeHeader, AgenticMooncakeRow,
            AgenticSnapshotOptions, AgenticSourceProvenance, PreparedAgenticSnapshots,
            WorkloadDriver, AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION,
        };

        let mut builder = AgenticGraphBuilder::new(AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 64,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "public-api-fixture".into(),
                digest: "authored-snapshot".into(),
            },
        })
        .unwrap();
        for (request_id, start) in [("before", 1_000.0), ("after", 1_100.0)] {
            builder
                .push(AgenticMooncakeRow {
                    request_id: request_id.into(),
                    play_id: "play".into(),
                    session_id: "session".into(),
                    model: "fixture-model".into(),
                    input_length: Some(128),
                    output_length: Some(1),
                    hash_ids: Some(vec![90, 10]),
                    not_before_ms: start,
                    recorded_api_time_ms: Some(10.0),
                    ..AgenticMooncakeRow::default()
                })
                .unwrap();
        }
        let graph = builder.finish().unwrap();
        let prepared = graph
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
            .unwrap();
        assert_eq!(prepared.snapshots().len(), 1);
        let evidence = &prepared.snapshots()[0];
        assert_eq!(evidence.seed, 42);
        assert!((1_025.0..1_075.0).contains(&evidence.t_star_ms));
        let context = prepared.context();
        assert!(context.prepare_play(0, 0, Some(0.0)).is_err());
        let first = context.prepare_play_from_start(0, 0).unwrap();
        let recycled = context.prepare_play_from_start(0, 1).unwrap();
        assert_eq!(first.evidence().t_star_ms, 1_000.0);
        assert!(first
            .evidence()
            .requests
            .iter()
            .all(|request| !request.historical));
        assert_ne!(first.evidence().cache_id, recycled.evidence().cache_id);
        assert_ne!(
            first.materialize_prefix("before", 128).unwrap(),
            recycled.materialize_prefix("before", 128).unwrap()
        );
        assert_eq!(
            first.materialize_prefix("before", 65).unwrap(),
            first.materialize_prefix("after", 128).unwrap()[..65]
        );
        let warmup = WorkloadDriver::new_agentic_warmup(prepared.clone(), 32, true, 2.0).unwrap();
        let phases = warmup.agentic_phase_evidence().unwrap();
        assert!(warmup.is_agentic_preparing());
        assert_eq!(phases.lanes[0].primers_expected, 1);
        assert_eq!(phases.lanes[0].warmup_expected, 10);
        assert_eq!(phases.requests[0].source_request_id, "before");
        assert!(phases
            .requests
            .iter()
            .all(|request| request.max_output_tokens == 1));
        assert_eq!(phases.profile_start_ms, None);
        // An external caller can run the same prepared context through the
        // public offline P/D executor without private runtime constructors.
        use aisimulate_core::replay::{
            ReplayEngineConfig, ReplayEngineFactory, ReplayRuntimeInput, ReplaySpec,
            ReplayTopology, Replayer, WorkerPoolSpec,
        };
        let expected = prepared.snapshots()[0]
            .requests
            .iter()
            .find(|request| request.source_request_id == "after")
            .unwrap()
            .identity
            .clone();
        let driver = WorkloadDriver::new_agentic_warmup(
            prepared,
            ReplayEngineConfig::default().rank.block_size,
            true,
            2.0,
        )
        .unwrap();
        let spec = ReplaySpec {
            version: 1,
            topology: ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec {
                    initial_workers: 1,
                    startup_delay_ms: 0.0,
                },
                decode: WorkerPoolSpec {
                    initial_workers: 1,
                    startup_delay_ms: 0.0,
                },
                handoff_latency_ms: 1.0,
            },
            engine: Default::default(),
            adapters: Default::default(),
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests: Vec::new(),
        };
        let report = Replayer::new(
            spec,
            ReplayEngineFactory::with_timing_model(std::sync::Arc::new(LatencyOnlyProvider)),
        )
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .run()
        .unwrap();
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request[0].agentic.as_ref(), Some(&expected));
        let phases = report.agentic_phases.unwrap();
        assert_eq!(phases.lanes[0].warmup_completed, 10);
        assert!(phases.profile_start_ms.is_some());
        let mut driver = WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![first]).unwrap(),
            32,
            true,
            2.0,
        )
        .unwrap();
        assert_eq!(driver.total_turns(), 2);
        assert_eq!(driver.next_ready_time_ms(), Some(0.0));
    }
}

/// Exhaustive downstream literals for the coordinated 0.13 report migration.
/// Cold callers explicitly leave both newly introduced phase fields absent.
/// Keeping this in an external crate catches public-field source changes.
pub fn rebuild_replay_report_literals(
    report: aisimulate_core::ReplayReport,
    record: aisimulate_core::replay::PerRequestRecord,
) -> (
    aisimulate_core::ReplayReport,
    aisimulate_core::replay::PerRequestRecord,
) {
    use aisimulate_core::{replay::PerRequestRecord, ReplayReport};
    (
        ReplayReport {
            kv_eviction_policy: report.kv_eviction_policy,
            kv_eviction_assumption: report.kv_eviction_assumption,
            committed_prefill_tokens: report.committed_prefill_tokens,
            g3_offload: report.g3_offload,
            request_counts: report.request_counts,
            throughput: report.throughput,
            prefix_cache_reused_ratio: report.prefix_cache_reused_ratio,
            first_admission_prefix_cache_reused_ratio: report
                .first_admission_prefix_cache_reused_ratio,
            latency: report.latency,
            trajectories: report.trajectories,
            agentic_graph: report.agentic_graph,
            agentic_snapshots: report.agentic_snapshots,
            agentic_phases: None,
            agentic_lifecycle: report.agentic_lifecycle,
            agentic_play_outcomes: report.agentic_play_outcomes,
            goodput: report.goodput,
            power: report.power,
            power_diagnostics: report.power_diagnostics,
            per_request: report.per_request,
            runtime_evidence: report.runtime_evidence,
        },
        PerRequestRecord {
            request_id: record.request_id,
            play_id: record.play_id,
            session_id: record.session_id,
            turn_index: record.turn_index,
            metadata: record.metadata,
            agentic: record.agentic,
            agentic_phase: None,
            uuid: record.uuid,
            arrival_time_ms: record.arrival_time_ms,
            dispatched_at_ms: record.dispatched_at_ms,
            first_admit_ms: record.first_admit_ms,
            terminal_time_ms: record.terminal_time_ms,
            first_token_ms: record.first_token_ms,
            last_token_ms: record.last_token_ms,
            ttft_ms: record.ttft_ms,
            ttst_ms: record.ttst_ms,
            e2e_latency_ms: record.e2e_latency_ms,
            itl_ms: record.itl_ms,
            input_length: record.input_length,
            requested_output_length: record.requested_output_length,
            output_length: record.output_length,
            reused_input_tokens: record.reused_input_tokens,
            first_admission_g1_reused_input_tokens: record.first_admission_g1_reused_input_tokens,
            first_admission_host_reused_input_tokens: record
                .first_admission_host_reused_input_tokens,
            prefill_worker_idx: record.prefill_worker_idx,
            decode_worker_idx: record.decode_worker_idx,
            prefill_admit_ms: record.prefill_admit_ms,
            source_held_ms: record.source_held_ms,
            destination_reserved_ms: record.destination_reserved_ms,
            destination_activated_ms: record.destination_activated_ms,
            decode_admit_ms: record.decode_admit_ms,
            source_released_ms: record.source_released_ms,
            decode_reused_input_tokens: record.decode_reused_input_tokens,
            prefill_route_overlap_tokens: record.prefill_route_overlap_tokens,
            decode_route_overlap_tokens: record.decode_route_overlap_tokens,
            routing_history: record.routing_history,
            admission_history: record.admission_history,
            admission_count: record.admission_count,
            readmission_count: record.readmission_count,
            pressure_record_ordinals: record.pressure_record_ordinals,
            terminal_status: record.terminal_status,
        },
    )
}

/// Detailed phase evidence is reachable through the canonical model.
pub fn operation_diagnostics(model: &ForwardPassPerfModel) -> Result<Vec<aisimulate_core::perfmodel::engine::diagnostics::StaticOperationDiagnostics>, AicError> {
    model.static_phase_diagnostics(1, 128, 0, true)
}
