// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit calibrated-silicon proof, never a fixed-model or availability skip.
//! Python initializes the canonical model and memory metadata only; actual
//! forward queries, scheduling and AgentX execution stay in the Rust process.

#[path = "support/run_fixture.rs"]
mod run_fixture;

use std::{collections::{BTreeMap,BTreeSet},path::PathBuf,rc::Rc};
use aisimulate_core::{BackendKind,DatabaseMode,perfmodel::{ForwardPassPerfModelConfig,ForwardPassWorkerType,EstimationMode,ForwardPassFallbackPolicy,memory::{KvCacheEstimateRequest,KvCacheEstimateOptions,KvCacheMemoryFraction,EstimateSource}}};
use aisimulate_core::engine::{CanonicalTimingModel,EngineConfig,EngineFactory,TimingModelConfig};

#[test]
fn h100_measured_op_tables_drive_native_agentx_with_shape_dependent_passes() {
    let systems = PathBuf::from(std::env::var_os("AISIM_SYSTEMS_ROOT").expect("AISIM_SYSTEMS_ROOT must identify the pinned checked-in systems assets; no skip or synthetic fallback"));
    assert!(systems.join("h100_sxm.yaml").is_file(),"H100 hardware metadata is required");
    let mut config = ForwardPassPerfModelConfig::new("meta-llama/Meta-Llama-3.1-8B","h100_sxm",BackendKind::Vllm,ForwardPassWorkerType::Aggregated);
    config.backend_version = Some("0.24.0".into());
    config.tp = 1; config.pp = 1; config.attention_dp = 1;
    config.gemm_quant_mode = Some("bfloat16".into());
    config.fmha_quant_mode = Some("bfloat16".into());
    config.kvcache_quant_mode = Some("bfloat16".into());
    config.estimation_mode = EstimationMode::OpLevel;
    config.fallback_policy = ForwardPassFallbackPolicy::Deny;
    config.database_mode = DatabaseMode::Silicon;
    config.strict_provenance = true;
    config.systems_paths = vec![systems.clone()];
    config.estimator_config.correction.enabled = false;
    let memory_engine = serde_json::from_value(serde_json::json!({
        "schema_version":aisimulate_core::perfmodel::ENGINE_CONFIG_SCHEMA_VERSION,
        "model_name":config.model,"system_name":config.system,
        "systems_path":systems,"backend":"vllm","backend_version":"0.24.0","forward_model":"op_level",
        "tp_size":1,"pp_size":1,"attention_dp_size":1,"kv_block_size":16,
        "weight_dtype":"bfloat16","activation_dtype":"bfloat16","kv_cache_dtype":"bfloat16",
        "strict_provenance":true,"database_mode":"silicon"
    })).expect("complete matching native memory engine identity");
    let (timing,memory) = CanonicalTimingModel::new_with_geometry(config,KvCacheEstimateRequest {
        engine:memory_engine,max_num_tokens:4096,max_batch_size:8,
        kv_cache_memory_fraction:KvCacheMemoryFraction::OfTotal(0.9),
        gpu_memory_capacity_bytes_override:None,cuda_graph_reserved_bytes:0,tolerance_fraction:Some(0.05),
        options:KvCacheEstimateOptions {allow_naive_fallback:false,allow_hf_config_download:false,naive_kv_reservation:0.8},
    }).expect("native calibrated model metadata must derive KV geometry without naive fallback");
    let provenance = timing.provenance().expect("canonical construction must retain provenance");
    assert_eq!(provenance.selected_estimation_mode,EstimationMode::OpLevel);
    assert_eq!(provenance.config.database_mode,DatabaseMode::Silicon);
    assert_eq!(provenance.config.fallback_policy,ForwardPassFallbackPolicy::Deny);
    assert!(provenance.selection_failures.is_empty(),"no alternate estimator can qualify this proof");
    assert_eq!(memory.source,EstimateSource::Native);
    assert_eq!(memory.total_gpu_capacity_bytes,85_899_345_920,"H10080GiB metadata identity");
    assert!(memory.kv_size_per_token_bytes > 0 && memory.total_kv_size_tokens > 0);
    let block_size = 16;
    let engine = EngineConfig {
        timing_model:TimingModelConfig::External {provider:"canonical_h100_op_level".into(),config:serde_json::Value::Null},
        max_num_seqs:8,max_num_batched_tokens:4096,
        block_size,num_gpu_blocks:usize::try_from(memory.tolerance_adjusted.as_ref().expect("explicit tolerance").total_kv_size_tokens).unwrap()/block_size,
        kv_cache_bytes_per_token:Some(usize::try_from(memory.kv_size_per_token_bytes).unwrap()),
        enable_prefix_caching:true,enable_chunked_prefill:true,async_scheduling:Some(false),
        ..Default::default()
    };
    let factory = EngineFactory::with_timing_model(engine,timing.clone()).expect("calibrated timing and native KV geometry must be admitted");
    let directory = tempfile::tempdir().unwrap();
    let raw = run_fixture::branch_request(directory.path());
    let mut request = aiperf_protocol::runner_v2::decode_run_request_v2(raw.as_bytes()).unwrap();
    request.run.cfg.models.as_mut().unwrap().items[0].name = "meta-llama/Meta-Llama-3.1-8B".into();
    let report = aisim_aiperf_workload_runner::execute_with_engine_factory(&request.run,aiperf_clock_runtime::cancellation::RunCancellation::new(),Rc::new(aiperf_timing::NoopPhaseObserver),&factory,8).expect("native AgentX must execute against measured H100 predictor");
    let dispatched = report["requests"].as_array().unwrap();
    assert_eq!(dispatched.len(),4);
    assert!(dispatched.iter().all(|row|row["request"]["body"]["model"] == provenance.config.model),"every native request body must address the calibrated model, not the source trace placeholder");
    let prompt_shapes:BTreeSet<usize> = dispatched.iter().map(|row|serde_json::to_vec(&row["request"]["body"]["messages"]).unwrap().len()).collect();
    assert!(prompt_shapes.len()>1,"actual native prompt shapes must vary, not only request counts");
    assert_eq!(report["engine_drained"],true);
    let mut starts = BTreeMap::new();
    let mut durations = BTreeSet::new();
    for event in report["model_trace"].as_array().unwrap() {
        if event["event"] == "pass_started" { starts.insert(event["pass_id"].as_u64().unwrap(),event["at_ns"].as_i64().unwrap()); }
        if event["event"] == "pass_completed" {
            let start = starts[&event["pass_id"].as_u64().unwrap()];
            durations.insert(event["at_ns"].as_i64().unwrap()-start);
        }
    }
    assert!(durations.len() > 1,"calibrated prefill/decode shapes must produce nonconstant measured-model pass durations: {durations:?}");
    assert!(durations.iter().all(|duration|*duration>0));
    let diagnostics = timing.diagnostics();
    assert_eq!(serde_json::to_value(&diagnostics).unwrap()["source"],"aic");
    println!("{}",serde_json::json!({"proof":"calibrated-h100-op-level-not-live-gpu-execution","aisim_source":env!("AISIM_SOURCE_REVISION"),"aiperf_source":env!("AIPERF_SOURCE_REVISION"),"native_model":diagnostics,"native_kv_geometry":memory,"actual_pass_durations_ns":durations,"native_agentx":report}));
}
