// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{EngineConfig, NativeHostOffloadConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ProviderSpec, ReplayAdapters, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayRequest, ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec,
    run_engine_replay,
};

fn engine_config() -> ReplayEngineConfig {
    ReplayEngineConfig {
        dp_size: 1,
        tensor_parallel_size: 1,
        cache_domain_ids: Vec::new(),
        rank: EngineConfig {
            num_gpu_blocks: 16,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 0.0,
                decode_ms: 0.0,
            },
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
        requests: Vec::new(),
    }
}

fn request(id: &str, arrival_time_ms: f64, tokens: Vec<u32>) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens: tokens.len(),
        input_token_ids: Some(tokens),
        output_tokens: 0,
        output_token_ids: None,
        dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

#[test]
fn attention_dp_native_host_offload_requires_cache_domains() {
    let mut config = engine_config();
    config.rank.kv_cache_bytes_per_token = Some(1);
    config.rank.native_host_offload =
        Some(NativeHostOffloadConfig::new(1).with_bandwidths(1.0, 1.0));

    let mut attention_dp = config.clone();
    attention_dp.dp_size = 2;
    let error = run_engine_replay(spec(attention_dp)).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("cache_domain_id"), "{message}");

    config.dp_size = 2;
    config.cache_domain_ids = vec![0, 0];
    let mut disaggregated = spec(config);
    disaggregated.topology = ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 0.0,
    };
    let mut replay_request = request("disagg-attention-dp", 0.0, vec![1, 2, 3, 4]);
    replay_request.output_tokens = 1;
    disaggregated.requests.push(replay_request);
    let report = run_engine_replay(disaggregated).unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);
}

#[test]
fn attention_dp_host_cache_is_shared_only_within_configured_domains() {
    for target_rank in 1..8 {
        let mut config = engine_config();
        config.dp_size = 8;
        config.cache_domain_ids = vec![10, 10, 10, 10, 20, 20, 20, 20];
        config.rank.num_gpu_blocks = 1;
        config.rank.max_num_seqs = 1;
        config.rank.max_num_batched_tokens = 4;
        config.rank.kv_cache_bytes_per_token = Some(250_000);
        config.rank.native_host_offload =
            Some(NativeHostOffloadConfig::new(8).with_bandwidths(1.0, 1.0));

        let mut replay = spec(config);
        let mut seed = request("turn-0-rank-0", 0.0, vec![1, 2, 3, 4]);
        seed.dp_rank = Some(0);
        seed.session_id = Some("session-a".to_string());
        seed.turn_index = Some(0);
        let mut evict = request("evict-rank-0", 3.0, vec![5, 6, 7, 8]);
        evict.dp_rank = Some(0);
        let probe_id = format!("turn-1-rank-{target_rank}");
        let mut probe = request(&probe_id, 6.0, vec![1, 2, 3, 4]);
        probe.dp_rank = Some(target_rank);
        probe.session_id = Some("session-a".to_string());
        probe.turn_index = Some(1);
        replay.requests = vec![seed, evict, probe];

        let report = Replayer::new(replay, ReplayEngineFactory::new())
            .unwrap()
            .with_capture_options(ReplayCaptureOptions {
                determinism: ReplayDeterminism::CanonicalV1,
                ..ReplayCaptureOptions::default()
            })
            .run()
            .unwrap();
        let probe = report
            .per_request
            .iter()
            .find(|record| record.request_id.as_deref() == Some(probe_id.as_str()))
            .unwrap();
        assert_eq!(
            probe.routing_history.last().unwrap().dp_rank,
            Some(target_rank)
        );
        assert_eq!(probe.first_admission_g1_reused_input_tokens, Some(0));
        if target_rank < 4 {
            assert_eq!(probe.reused_input_tokens, 4, "rank {target_rank}");
            assert_eq!(
                probe.first_admission_host_reused_input_tokens,
                Some(4),
                "rank {target_rank}"
            );
        } else {
            assert_eq!(probe.reused_input_tokens, 0, "rank {target_rank}");
            assert_eq!(
                probe.first_admission_host_reused_input_tokens,
                Some(0),
                "rank {target_rank}"
            );
        }
    }
}
