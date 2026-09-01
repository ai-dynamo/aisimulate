// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use aisimulate_core::engine::{
    Backend, EngineConfig, NativeHostOffloadConfig, SglangConfig, TimingModel, TimingModelConfig,
};
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, PoolRoundRobinPlacement,
    ProviderSpec, ReplayAdapters, ReplayCaptureOptions, ReplayComposition, ReplayDeterminism,
    ReplayEngineConfig, ReplayEngineFactory, ReplayReport, ReplayRequest, ReplayRoleConfig,
    ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec, ReplayTopology,
    Replayer, WorkerPoolSpec, WorkerStage, WorkerTopology, run_engine_replay,
    run_engine_replay_with_optional_role_timing, run_engine_replay_with_timing,
};
use anyhow::Result;

fn assert_deterministic(first: &ReplayReport, second: &ReplayReport) {
    let mut first = first.clone();
    let mut second = second.clone();
    first.throughput.wall_time_ms = 0.0;
    second.throughput.wall_time_ms = 0.0;
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&second).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&first.per_request).unwrap(),
        serde_json::to_value(&second.per_request).unwrap()
    );
}

fn run_canonical_engine_replay(spec: ReplaySpec) -> ReplayReport {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(ReplayCaptureOptions {
            determinism: ReplayDeterminism::CanonicalV1,
            ..ReplayCaptureOptions::default()
        })
        .run()
        .unwrap()
}

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens,
        input_token_ids: None,
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

fn request_with_tokens(
    id: &str,
    arrival_time_ms: f64,
    input_token_ids: Vec<u32>,
    output_tokens: usize,
) -> ReplayRequest {
    let mut request = request(id, arrival_time_ms, input_token_ids.len(), output_tokens);
    request.input_token_ids = Some(input_token_ids);
    request
}

fn native_host_offload_config(
    host_capacity_blocks: usize,
    prefill_ms: f64,
    decode_ms: f64,
) -> ReplayEngineConfig {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms,
        decode_ms,
    });
    config.rank.num_gpu_blocks = 2;
    config.rank.max_num_seqs = 2;
    config.rank.max_num_batched_tokens = 8;
    config.rank.kv_cache_bytes_per_token = Some(250_000);
    config.rank.native_host_offload =
        Some(NativeHostOffloadConfig::new(host_capacity_blocks).with_bandwidths(1.0, 1.0));
    config
}

fn engine_config(timing_model: TimingModelConfig) -> ReplayEngineConfig {
    ReplayEngineConfig {
        dp_size: 1,
        tensor_parallel_size: 1,
        rank: EngineConfig {
            num_gpu_blocks: 16,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model,
            ..EngineConfig::default()
        },
        ..ReplayEngineConfig::default()
    }
}

fn spec(config: ReplayEngineConfig) -> ReplaySpec {
    ReplaySpec {
        version: 1,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 1,
                startup_delay_ms: 0.0,
            },
        },
        engine: serde_json::to_value(config).unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: vec![request("request-a", 0.0, 4, 2)],
    }
}

fn role_config(backend: Backend, timing_model: TimingModelConfig) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: 1,
        tensor_parallel_size: 1,
        cache_domain_ids: Vec::new(),
        rank: EngineConfig {
            num_gpu_blocks: 32,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model,
            ..EngineConfig::for_backend(backend)
        },
    }
}

fn disaggregated_spec(
    backend: Backend,
    prefill_timing: TimingModelConfig,
    decode_timing: TimingModelConfig,
) -> ReplaySpec {
    let mut spec = spec(ReplayEngineConfig {
        prefill: Some(role_config(backend, prefill_timing)),
        decode: Some(role_config(backend, decode_timing)),
        ..ReplayEngineConfig::default()
    });
    spec.topology = ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 1.0,
    };
    spec
}

#[test]
fn disaggregated_replay_rejects_attention_dp_before_engine_materialization() {
    for stage in [WorkerStage::Prefill, WorkerStage::Decode] {
        let mut spec = disaggregated_spec(
            Backend::Vllm,
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
        );
        let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
        match stage {
            WorkerStage::Prefill => config.prefill.as_mut().unwrap().dp_size = 2,
            WorkerStage::Decode => config.decode.as_mut().unwrap().dp_size = 2,
            WorkerStage::Aggregated => unreachable!(),
        }
        spec.engine = serde_json::to_value(config).unwrap();

        let error = run_engine_replay(spec).unwrap_err();
        assert!(matches!(
            error,
            aisimulate_core::replay::ReplayError::InvalidSpec(_)
        ));
        let role_name = match stage {
            WorkerStage::Prefill => "prefill",
            WorkerStage::Decode => "decode",
            WorkerStage::Aggregated => unreachable!(),
        };
        assert!(
            error
                .to_string()
                .contains(&format!("{role_name} dp_size=1"))
        );
    }
}

#[test]
fn native_execution_descriptor_round_trips_external_provider_config() {
    let config = engine_config(TimingModelConfig::External {
        provider: "aic".to_string(),
        config: serde_json::json!({
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "backend": "vllm",
            "system": "h100_sxm",
            "tp": 2,
        }),
    });
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(
        serde_json::from_value::<ReplayEngineConfig>(value).unwrap(),
        config
    );
}

#[test]
fn built_in_aggregated_replay_produces_a_deterministic_report() {
    let spec = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 10.0,
        decode_ms: 2.0,
    }));
    let first = run_canonical_engine_replay(spec.clone());
    let second = run_canonical_engine_replay(spec);
    assert_deterministic(&first, &second);
    assert_eq!(first.request_counts.completed_requests, 1);
    assert_eq!(first.request_counts.total_input_tokens, 4);
    assert_eq!(first.request_counts.total_output_tokens, 2);
    assert_eq!(first.throughput.duration_ms, 14.0);
    assert_eq!(first.throughput.decode_gpus_per_worker, 1);
    assert_eq!(first.per_request[0].first_token_ms, Some(12.0));
    assert_eq!(first.per_request[0].terminal_time_ms, 14.0);
}

#[test]
fn vllm_native_host_offload_restores_an_evicted_prefix_through_internal_work() {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 0.0,
        decode_ms: 0.0,
    });
    config.rank.num_gpu_blocks = 1;
    config.rank.max_num_seqs = 1;
    config.rank.max_num_batched_tokens = 4;
    config.rank.kv_cache_bytes_per_token = Some(250_000);
    config.rank.native_host_offload =
        Some(NativeHostOffloadConfig::new(2).with_bandwidths(1.0, 1.0));

    let mut replay = spec(config);
    replay.requests = vec![
        request("seed", 0.0, 4, 0),
        request("evict", 5.0, 4, 0),
        request("restore", 10.0, 4, 0),
    ];
    replay.requests[0].input_token_ids = Some(vec![1, 2, 3, 4]);
    replay.requests[1].input_token_ids = Some(vec![5, 6, 7, 8]);
    replay.requests[2].input_token_ids = Some(vec![1, 2, 3, 4]);

    let report = run_canonical_engine_replay(replay);
    assert_eq!(report.request_counts.completed_requests, 3);
    let restore = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("restore"))
        .unwrap();
    assert_eq!(restore.reused_input_tokens, 4);
    assert_eq!(restore.first_admission_g1_reused_input_tokens, Some(0));
    assert_eq!(restore.first_admission_host_reused_input_tokens, Some(4));
    assert!(restore.first_admit_ms.is_some_and(|at_ms| at_ms >= 11.0));
    assert_eq!(restore.admission_history.len(), 1);
    assert_eq!(restore.admission_history[0].g1_reused_input_tokens, Some(0));
    assert_eq!(
        restore.admission_history[0].host_reused_input_tokens,
        Some(4)
    );
}

#[test]
fn host_load_due_during_a_forward_pass_waits_for_the_pass_boundary() {
    let mut replay = spec(native_host_offload_config(4, 5.0, 0.0));
    replay.requests = vec![
        request_with_tokens("seed", 0.0, vec![1, 2, 3, 4], 0),
        // Fill both G1 slots so `seed` survives only in the host tier.
        request_with_tokens("evict", 10.0, vec![5, 6, 7, 8, 9, 10, 11, 12], 0),
        // This queues a one-block H2D at 20ms, due at 21ms.
        request_with_tokens("restore", 20.0, vec![1, 2, 3, 4], 0),
        // Keep the engine in a nonzero forward pass through 25ms.
        request_with_tokens("busy", 20.0, vec![13, 14, 15, 16], 0),
    ];

    let report = run_canonical_engine_replay(replay);
    let restore = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("restore"))
        .unwrap();
    let busy = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("busy"))
        .unwrap();

    assert_eq!(busy.first_admit_ms, Some(20.0));
    assert_eq!(busy.terminal_time_ms, 25.0);
    assert_eq!(restore.first_admit_ms, Some(25.0));
    assert_eq!(restore.first_admission_g1_reused_input_tokens, Some(0));
    assert_eq!(restore.first_admission_host_reused_input_tokens, Some(4));
}

#[test]
fn activated_host_request_precedes_a_newly_preempted_normal_request() {
    let mut replay = spec(native_host_offload_config(1, 5.0, 1.0));
    replay.requests = vec![
        request_with_tokens("seed", 0.0, vec![1, 2, 3, 4], 0),
        // Evict `seed` from G1 while its one-block host copy remains resident.
        request_with_tokens("evict-seed", 10.0, vec![5, 6, 7, 8, 9, 10, 11, 12], 0),
        // H2D reserves one of two G1 blocks. `source` initially fits in the
        // other block, but its first decode growth requires another block and
        // preempts it after the host load has activated.
        request_with_tokens("load-seed", 20.0, vec![1, 2, 3, 4], 0),
        request_with_tokens("source", 20.0, vec![21, 22, 23, 24], 3),
    ];

    let report = run_engine_replay(replay).expect(
        "the activated host request must run before the newly preempted normal request so its G1 reservation can be released",
    );
    assert_eq!(report.request_counts.completed_requests, 4);
    let loaded = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("load-seed"))
        .unwrap();
    assert_eq!(loaded.first_admission_g1_reused_input_tokens, Some(0));
    assert_eq!(loaded.first_admission_host_reused_input_tokens, Some(4));
}

#[test]
fn host_store_capacity_retry_preserves_the_request_cursor() {
    let mut config = native_host_offload_config(1, 5.0, 1.0);
    // H2D owns one block while `source` owns one complete prompt block plus a
    // partial tail. Three G1 slots isolate host-store retry from decode OOM.
    config.rank.num_gpu_blocks = 3;
    let mut replay = spec(config);
    replay.requests = vec![
        request_with_tokens("seed", 0.0, vec![1, 2, 3, 4], 0),
        // Evict `seed` from G1. This three-block cohort cannot replace it in the
        // one-block host tier, so the host still contains only `seed`.
        request_with_tokens(
            "evict-seed",
            10.0,
            vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            0,
        ),
        // The H2D pins the only host block while `source` first completes its
        // prompt block, forcing that store attempt to retry without advancing.
        request_with_tokens("load-seed", 20.0, vec![1, 2, 3, 4], 0),
        request_with_tokens("source", 20.0, vec![21, 22, 23, 24, 25], 3),
        // After `source` finishes, evict its G1 block. This cohort is also too
        // large for G2 and therefore cannot hide a missing cursor retry.
        request_with_tokens(
            "evict-source",
            40.0,
            vec![31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            0,
        ),
        request_with_tokens("restore-source", 50.0, vec![21, 22, 23, 24], 0),
    ];

    let report = run_canonical_engine_replay(replay);
    let restored = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("restore-source"))
        .unwrap();

    assert_eq!(report.request_counts.completed_requests, 6);
    assert_eq!(restored.reused_input_tokens, 4);
    assert_eq!(restored.first_admission_g1_reused_input_tokens, Some(0));
    assert_eq!(restored.first_admission_host_reused_input_tokens, Some(4));
    assert!(restored.first_admit_ms.is_some_and(|at_ms| at_ms >= 51.0));
}

#[test]
fn replay_report_retains_authored_request_correlation() {
    let mut replay = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    }));
    replay.requests[0].id = "caller-request-17".into();
    replay.requests[0].input_token_ids = Some(vec![10, 11, 12, 13]);
    replay.requests[0].session_id = Some("conversation-a".into());
    replay.requests[0].turn_index = Some(3);
    replay.requests[0].metadata = serde_json::json!({
        "priority": 5,
        "strict_priority": 2,
        "policy_class": "interactive",
        "caller_tag": "round-trip"
    });

    let report = run_engine_replay(replay).unwrap();
    let record = &report.per_request[0];
    assert_eq!(record.request_id.as_deref(), Some("caller-request-17"));
    assert_eq!(record.session_id.as_deref(), Some("conversation-a"));
    assert_eq!(record.turn_index, Some(3));
    assert_eq!(record.metadata["caller_tag"], "round-trip");
    assert_eq!(record.metadata["priority"], 5);
}

#[test]
fn multi_worker_round_robin_uses_each_logical_worker_deterministically() {
    let mut replay = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    }));
    replay.topology = ReplayTopology::Aggregated {
        workers: WorkerPoolSpec {
            initial_workers: 2,
            startup_delay_ms: 0.0,
        },
    };
    replay.requests = (0..4)
        .map(|index| request(&format!("rr-{index}"), 0.0, 4, 1))
        .collect();

    let first = run_canonical_engine_replay(replay.clone());
    let second = run_canonical_engine_replay(replay);
    assert_deterministic(&first, &second);
    let mut workers = first
        .per_request
        .iter()
        .map(|record| record.decode_worker_idx.unwrap())
        .collect::<Vec<_>>();
    workers.sort_unstable();
    assert_eq!(workers, vec![0, 0, 1, 1]);
}

struct ScaleOnce {
    fired: bool,
}

impl ReplayScalingPolicy for ScaleOnce {
    fn initial_tick_ms(&mut self) -> Result<f64> {
        Ok(0.0)
    }

    fn on_tick(&mut self, _snapshot: ReplayScalingSnapshot) -> Result<ReplayScalingDecision> {
        assert!(!self.fired, "one-shot policy must not be called twice");
        self.fired = true;
        Ok(ReplayScalingDecision {
            target_decode: Some(2),
            next_tick_ms: None,
            ..Default::default()
        })
    }
}

struct ScalingRoundRobin {
    policy: Option<Box<dyn ReplayScalingPolicy>>,
}

impl ReplayComposition for ScalingRoundRobin {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.policy.take())
    }
}

#[test]
fn scaling_composition_changes_round_robin_capacity_before_arrival() {
    let mut replay = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    }));
    replay.adapters.scaling = ProviderSpec {
        provider: "test_scaler".to_string(),
        config: serde_json::Value::Null,
    };
    replay.requests = (0..4)
        .map(|index| request(&format!("scaled-{index}"), 10.0, 4, 1))
        .collect();
    let composition = ScalingRoundRobin {
        policy: Some(Box::new(ScaleOnce { fired: false })),
    };
    let report = Replayer::with_composition(
        replay,
        aisimulate_core::replay::ReplayEngineFactory::new(),
        composition,
    )
    .unwrap()
    .run()
    .unwrap();

    let mut workers = report
        .per_request
        .iter()
        .map(|record| record.decode_worker_idx.unwrap())
        .collect::<Vec<_>>();
    workers.sort_unstable();
    assert_eq!(workers, vec![0, 0, 1, 1]);
}

struct CaptureAttentionDpFpm {
    identities: Rc<RefCell<Vec<(usize, String, u32)>>>,
}

impl ReplayScalingPolicy for CaptureAttentionDpFpm {
    fn initial_tick_ms(&mut self) -> Result<f64> {
        Ok(250.0)
    }

    fn on_tick(&mut self, snapshot: ReplayScalingSnapshot) -> Result<ReplayScalingDecision> {
        self.identities.borrow_mut().extend(
            snapshot
                .decode_fpm
                .into_iter()
                .map(|(worker_id, fpm)| (worker_id, fpm.worker_id, fpm.dp_rank)),
        );
        Ok(ReplayScalingDecision::default())
    }
}

#[test]
fn attention_dp_offline_fpm_preserves_logical_worker_and_rank_identity() {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 100.0,
        decode_ms: 100.0,
    });
    config.dp_size = 2;
    let mut replay = spec(config);
    replay.adapters.scaling = ProviderSpec {
        provider: "capture_attention_dp_fpm".to_string(),
        config: serde_json::Value::Null,
    };
    replay.requests = vec![
        ReplayRequest {
            dp_rank: Some(0),
            ..request("rank-0", 0.0, 8, 20)
        },
        ReplayRequest {
            dp_rank: Some(1),
            ..request("rank-1", 0.0, 8, 20)
        },
    ];
    let identities = Rc::new(RefCell::new(Vec::new()));
    let composition = ScalingRoundRobin {
        policy: Some(Box::new(CaptureAttentionDpFpm {
            identities: Rc::clone(&identities),
        })),
    };

    let report = Replayer::with_composition(replay, ReplayEngineFactory::new(), composition)
        .unwrap()
        .run()
        .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(
        *identities.borrow(),
        vec![(0, "0".to_string(), 0), (0, "0".to_string(), 1)]
    );
}

#[test]
fn engine_replay_preserves_exact_output_token_plan() {
    let mut replay = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    }));
    replay.requests[0].output_tokens = 1;
    replay.requests[0].output_token_ids = Some(vec![101, 102, 103]);

    let report = run_engine_replay(replay).unwrap();
    assert_eq!(report.request_counts.total_output_tokens, 3);
    assert_eq!(report.per_request[0].requested_output_length, 3);
    assert_eq!(report.per_request[0].output_length, 3);
}

#[test]
fn engine_replay_rejects_an_out_of_range_preassigned_dp_rank() {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    });
    config.dp_size = 2;
    let mut replay = spec(config);
    replay.requests[0].dp_rank = Some(2);

    let error = run_engine_replay(replay).unwrap_err();
    assert!(error.to_string().contains("DP rank 2"), "{error}");
}

fn impossible_sglang_config() -> ReplayEngineConfig {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    });
    config.rank = EngineConfig {
        backend: Backend::Sglang,
        num_gpu_blocks: 1,
        block_size: 4,
        sglang: SglangConfig {
            chunked_prefill_size: 8,
            ..SglangConfig::default()
        },
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 1.0,
            decode_ms: 1.0,
        },
        ..EngineConfig::for_backend(Backend::Sglang)
    };
    config
}

#[test]
fn impossible_sglang_request_returns_a_livelock_error_instead_of_spinning() {
    let mut replay = spec(impossible_sglang_config());
    replay.requests = vec![request("impossible", 0.0, 8, 2)];

    let error = run_engine_replay(replay).unwrap_err();
    assert_eq!(
        error.to_string(),
        "replay invariant violated: offline replay detected an effect-free zero-duration pass with 1 in-flight requests remaining"
    );
}

#[test]
fn impossible_sglang_request_cannot_escape_into_a_future_event_or_soft_cap() {
    let mut replay = spec(impossible_sglang_config());
    replay.max_sim_time_ms = Some(50.0);
    replay.requests = vec![
        request("impossible", 0.0, 8, 2),
        request("future", 100.0, 4, 1),
    ];

    let error = run_engine_replay(replay).unwrap_err();
    assert_eq!(
        error.to_string(),
        "replay invariant violated: offline replay detected an effect-free zero-duration pass with 1 in-flight requests remaining"
    );
}

#[test]
fn impossible_disaggregated_sglang_prefill_is_not_hidden_as_an_external_wait() {
    let fixed = TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    };
    let mut replay = disaggregated_spec(Backend::Sglang, fixed.clone(), fixed);
    let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
    let prefill = config.prefill.as_mut().unwrap();
    prefill.rank.num_gpu_blocks = 1;
    prefill.rank.block_size = 4;
    prefill.rank.sglang.chunked_prefill_size = 8;
    replay.engine = serde_json::to_value(config).unwrap();
    replay.requests = vec![request("impossible-disagg-prefill", 0.0, 8, 2)];

    let error = run_engine_replay(replay).unwrap_err();
    assert!(
        error.to_string().contains("effect-free zero-duration pass"),
        "{error}"
    );
}

#[test]
fn resource_accounting_multiplies_attention_dp_and_tensor_parallelism() {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    });
    config.dp_size = 2;
    config.tensor_parallel_size = 3;
    let report = run_engine_replay(spec(config)).unwrap();
    assert_eq!(report.throughput.decode_gpus_per_worker, 6);
}

struct FixedExternalTiming {
    prefill_ms: f64,
    decode_ms: f64,
}

impl TimingModel for FixedExternalTiming {
    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> Result<f64> {
        Ok(self.prefill_ms)
    }

    fn predict_decode_ms(
        &self,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        Ok(self.decode_ms)
    }
}

#[test]
fn runner_must_resolve_external_timing_before_execution() {
    let spec = spec(engine_config(TimingModelConfig::External {
        provider: "aic".to_string(),
        config: serde_json::json!({"model": "test"}),
    }));
    let error = run_engine_replay(spec.clone()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("timing provider 'aic' requires EngineFactory::with_timing_model"),
        "{error}"
    );

    let report = run_engine_replay_with_timing(
        spec,
        Arc::new(FixedExternalTiming {
            prefill_ms: 5.0,
            decode_ms: 1.0,
        }),
    )
    .unwrap();
    assert_eq!(report.throughput.duration_ms, 7.0);
}

#[test]
fn native_vllm_disaggregated_replay_completes_deterministically() {
    let spec = disaggregated_spec(
        Backend::Vllm,
        TimingModelConfig::Fixed {
            prefill_ms: 3.0,
            decode_ms: 1.0,
        },
        TimingModelConfig::Fixed {
            prefill_ms: 3.0,
            decode_ms: 2.0,
        },
    );
    let first = run_canonical_engine_replay(spec.clone());
    let second = run_canonical_engine_replay(spec);
    assert_deterministic(&first, &second);
    assert_eq!(first.request_counts.completed_requests, 1);
    assert_eq!(first.request_counts.total_input_tokens, 4);
    assert_eq!(first.request_counts.total_output_tokens, 2);
    assert_eq!(first.per_request[0].output_length, 2);
    let request = &first.per_request[0];
    assert_eq!(request.prefill_worker_idx, Some(0));
    assert_eq!(request.decode_worker_idx, Some(0));
    assert!(request.prefill_admit_ms.is_some());
    assert!(request.source_held_ms.is_some());
    assert!(request.destination_reserved_ms.is_some());
    assert!(request.destination_activated_ms.is_some());
    assert!(request.decode_admit_ms.is_some());
    assert!(request.source_released_ms.is_some());
    assert_eq!(request.decode_reused_input_tokens, Some(0));
    assert_eq!(request.prefill_route_overlap_tokens, Some(0));
    assert_eq!(request.decode_route_overlap_tokens, Some(0));
}

#[test]
fn disaggregated_handoff_latency_is_used_when_engine_timing_is_missing() {
    let fixed = TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    };
    let mut zero = disaggregated_spec(Backend::Vllm, fixed.clone(), fixed);
    if let ReplayTopology::Disaggregated {
        handoff_latency_ms, ..
    } = &mut zero.topology
    {
        *handoff_latency_ms = 0.0;
    }
    let mut fallback = zero.clone();
    if let ReplayTopology::Disaggregated {
        handoff_latency_ms, ..
    } = &mut fallback.topology
    {
        *handoff_latency_ms = 7.0;
    }

    let zero_report = run_engine_replay(zero).unwrap();
    let fallback_report = run_engine_replay(fallback).unwrap();
    assert_eq!(
        fallback_report.per_request[0].destination_activated_ms,
        zero_report.per_request[0]
            .destination_activated_ms
            .map(|time| time + 7.0)
    );
    assert_eq!(
        fallback_report.per_request[0].terminal_time_ms,
        zero_report.per_request[0].terminal_time_ms + 7.0
    );
}

#[test]
fn native_sglang_disaggregated_replay_completes_deterministically() {
    let spec = disaggregated_spec(
        Backend::Sglang,
        TimingModelConfig::Fixed {
            prefill_ms: 4.0,
            decode_ms: 1.0,
        },
        TimingModelConfig::Fixed {
            prefill_ms: 4.0,
            decode_ms: 2.0,
        },
    );
    let first = run_canonical_engine_replay(spec.clone());
    let second = run_canonical_engine_replay(spec);
    assert_deterministic(&first, &second);
    assert_eq!(first.request_counts.completed_requests, 1);
    assert_eq!(first.request_counts.total_input_tokens, 4);
    assert_eq!(first.request_counts.total_output_tokens, 2);
    assert_eq!(first.per_request[0].output_length, 2);
    let request = &first.per_request[0];
    assert_eq!(request.prefill_worker_idx, Some(0));
    assert_eq!(request.decode_worker_idx, Some(0));
    assert!(request.prefill_admit_ms.is_some());
    assert!(request.source_held_ms.is_some());
    assert!(request.destination_reserved_ms.is_some());
    assert!(request.destination_activated_ms.is_some());
    assert!(request.decode_admit_ms.is_some());
    assert!(request.source_released_ms.is_some());
    assert_eq!(request.decode_reused_input_tokens, Some(0));
    assert_eq!(request.prefill_route_overlap_tokens, Some(0));
    assert_eq!(request.decode_route_overlap_tokens, Some(0));
}

#[test]
fn disaggregated_prefill_does_not_reserve_or_generate_the_decode_output_length() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        let mut spec = disaggregated_spec(
            backend,
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
        );
        spec.requests = vec![request("long-decode", 0.0, 3, 8)];

        let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
        let prefill = config.prefill.as_mut().unwrap();
        prefill.rank.block_size = 4;
        prefill.rank.num_gpu_blocks = 2;
        let decode = config.decode.as_mut().unwrap();
        decode.rank.block_size = 4;
        decode.rank.num_gpu_blocks = 64;
        spec.engine = serde_json::to_value(config).unwrap();

        let report = run_engine_replay(spec)
            .unwrap_or_else(|error| panic!("{backend:?} disaggregated replay failed: {error}"));
        assert_eq!(report.request_counts.completed_requests, 1, "{backend:?}");
        assert_eq!(report.request_counts.total_output_tokens, 8, "{backend:?}");
    }
}

#[test]
fn role_specific_timing_models_support_external_and_builtin_mixes() {
    let external = TimingModelConfig::External {
        provider: "aic".to_string(),
        config: serde_json::json!({"model": "prefill"}),
    };
    let fixed = TimingModelConfig::Fixed {
        prefill_ms: 4.0,
        decode_ms: 2.0,
    };
    let spec = disaggregated_spec(Backend::Vllm, external.clone(), external);
    let report = run_engine_replay_with_optional_role_timing(
        spec,
        Some(Arc::new(FixedExternalTiming {
            prefill_ms: 3.0,
            decode_ms: 1.0,
        })),
        Some(Arc::new(FixedExternalTiming {
            prefill_ms: 4.0,
            decode_ms: 2.0,
        })),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);

    let spec = disaggregated_spec(
        Backend::Vllm,
        TimingModelConfig::External {
            provider: "aic".to_string(),
            config: serde_json::json!({"model": "prefill"}),
        },
        fixed,
    );
    let report = run_engine_replay_with_optional_role_timing(
        spec,
        Some(Arc::new(FixedExternalTiming {
            prefill_ms: 3.0,
            decode_ms: 1.0,
        })),
        None,
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);
}

#[test]
fn native_trtllm_disaggregated_replay_is_an_explicit_error() {
    let spec = disaggregated_spec(
        Backend::Trtllm,
        TimingModelConfig::Polynomial,
        TimingModelConfig::Polynomial,
    );
    let error = run_engine_replay(spec).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("offline disaggregated replay does not support TRT-LLM"),
        "{error}"
    );
}
