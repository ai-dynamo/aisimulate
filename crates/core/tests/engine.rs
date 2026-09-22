// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use aisimulate_core::engine::generalized::{EngineIdentity, SameTimestampRetry, SchedulerCommand};
use aisimulate_core::engine::{
    Backend, Command, Engine, EngineConfig, EngineFactory, NativeHostOffloadConfig,
    PassCompletionEffects, Request, SglangConfig, TimingModel, TimingModelConfig,
};
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, PoolRoundRobinPlacement,
    ProviderSpec, ReplayAdapters, ReplayCaptureOptions, ReplayComposition, ReplayDeterminism,
    ReplayEngineConfig, ReplayEngineFactory, ReplayReport, ReplayRequest, ReplayRequestPool,
    ReplayRoleConfig, ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot,
    ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec, WorkerTopology, run_engine_replay,
    run_engine_replay_with_optional_role_timing, run_engine_replay_with_timing,
};
use anyhow::Result;
use uuid::Uuid;

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
        prefill_dp_rank: None,
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

#[test]
fn every_backend_stops_at_max_model_len() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for speculative in [false, true] {
            for planned in [false, true] {
                for (prompt, output, expected) in [(3, 10, 5), (7, 10, 1), (3, 2, 2), (3, 0, 0)] {
                    let mut config = engine_config(TimingModelConfig::Fixed {
                        prefill_ms: 1.0,
                        decode_ms: 1.0,
                    });
                    config.rank.backend = backend;
                    config.rank.max_model_len = Some(8);
                    config.rank.sglang.chunked_prefill_size = 4;
                    if speculative {
                        config.rank.aic_nextn = Some(2);
                        config.rank.aic_nextn_accept_rates = Some("1,1".to_string());
                    }
                    let mut replay = spec(config);
                    let mut req = request("limited", 0.0, prompt, output);
                    if planned {
                        req.output_token_ids = Some((100..100 + output as u32).collect());
                    }
                    replay.requests = vec![req];
                    let report = run_engine_replay(replay).unwrap();
                    assert_eq!(report.request_counts.completed_requests, 1, "{backend:?}");
                    assert_eq!(
                        report.per_request[0].output_length, expected,
                        "{backend:?}, speculative={speculative}, planned={planned}, prompt={prompt}"
                    );
                    assert_eq!(report.request_counts.total_output_tokens, expected);
                    assert_eq!(report.per_request[0].requested_output_length, output);
                }
            }
        }
    }
}

#[rstest::rstest]
fn backends_reserve_only_context_capped_output(
    #[values(Backend::Trtllm, Backend::Sglang)] backend: Backend,
) {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    });
    config.rank.backend = backend;
    config.rank.max_model_len = Some(8);
    config.rank.num_gpu_blocks = 5;
    config.rank.enable_prefix_caching = false;
    let mut replay = spec(config);
    replay.requests = vec![request("a", 0.0, 4, 100), request("b", 0.0, 4, 100)];
    let report = run_engine_replay(replay).unwrap();
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_output_tokens, 8);
    assert_eq!(report.per_request[0].first_admit_ms, Some(0.0));
    assert_eq!(
        report.per_request[0].first_admit_ms,
        report.per_request[1].first_admit_ms
    );
}

#[test]
fn disaggregated_backends_stop_at_max_model_len() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for prompt in [3, 7] {
            let timing = TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            };
            let mut replay = disaggregated_spec(backend, timing.clone(), timing);
            let mut config: ReplayEngineConfig =
                serde_json::from_value(replay.engine.clone()).unwrap();
            config.prefill.as_mut().unwrap().rank.max_model_len = Some(8);
            config.decode.as_mut().unwrap().rank.max_model_len = Some(8);
            replay.engine = serde_json::to_value(config).unwrap();
            replay.requests = vec![request("limited", 0.0, prompt, 10)];
            let report = run_engine_replay(replay).unwrap();
            assert_eq!(report.request_counts.completed_requests, 1, "{backend:?}");
            assert_eq!(
                report.per_request[0].output_length,
                8 - prompt,
                "{backend:?}"
            );
        }
    }
}

fn role_config(backend: Backend, timing_model: TimingModelConfig) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: 1,
        tensor_parallel_size: 1,
        rank: EngineConfig {
            num_gpu_blocks: 32,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model,
            ..EngineConfig::for_backend(backend)
        },
        ..ReplayRoleConfig::default()
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
fn disaggregated_replay_supports_attention_dp_for_each_backend() {
    for backend in [Backend::Vllm, Backend::Sglang, Backend::Trtllm] {
        for (prefill_dp, decode_dp) in [(2, 1), (1, 2), (2, 4), (2, 2)] {
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
            let mut config: ReplayEngineConfig =
                serde_json::from_value(spec.engine.clone()).unwrap();
            config.prefill.as_mut().unwrap().dp_size = prefill_dp;
            config.decode.as_mut().unwrap().dp_size = decode_dp;
            spec.engine = serde_json::to_value(config).unwrap();
            spec.requests = (0..8)
                .map(|index| request(&format!("request-{index}"), 0.0, 4, 2))
                .collect();

            let report = run_engine_replay(spec).unwrap();
            assert_eq!(report.request_counts.completed_requests, 8);
            assert_eq!(report.per_request.len(), 8);
            for record in &report.per_request {
                let prefill = record
                    .routing_history
                    .iter()
                    .find(|route| route.pool == ReplayRequestPool::Prefill)
                    .unwrap();
                let decode = record
                    .routing_history
                    .iter()
                    .find(|route| route.pool == ReplayRequestPool::Decode)
                    .unwrap();
                assert!(prefill.dp_rank.unwrap() < prefill_dp);
                assert!(decode.dp_rank.unwrap() < decode_dp);
                assert_eq!(prefill.logical_worker_id, Some(0));
                assert_eq!(decode.logical_worker_id, Some(0));
            }
        }
    }
}

#[test]
fn disaggregated_replay_honors_authored_prefill_and_decode_dp_ranks() {
    let mut replay = disaggregated_spec(
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
    let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
    config.prefill.as_mut().unwrap().dp_size = 2;
    config.decode.as_mut().unwrap().dp_size = 4;
    replay.engine = serde_json::to_value(config).unwrap();
    replay.requests = vec![ReplayRequest {
        dp_rank: Some(3),
        prefill_dp_rank: Some(1),
        ..request("p1-to-d3", 0.0, 4, 2)
    }];

    let report = run_engine_replay(replay).unwrap();
    let record = &report.per_request[0];
    let prefill = record
        .routing_history
        .iter()
        .find(|route| route.pool == ReplayRequestPool::Prefill)
        .unwrap();
    let decode = record
        .routing_history
        .iter()
        .find(|route| route.pool == ReplayRequestPool::Decode)
        .unwrap();
    assert_eq!(prefill.dp_rank, Some(1));
    assert_eq!(decode.dp_rank, Some(3));
}

#[test]
fn disaggregated_replay_uses_decode_dp_rank_as_prefill_fallback() {
    let mut replay = disaggregated_spec(
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
    let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
    config.prefill.as_mut().unwrap().dp_size = 2;
    config.decode.as_mut().unwrap().dp_size = 2;
    replay.engine = serde_json::to_value(config).unwrap();
    replay.requests = vec![ReplayRequest {
        dp_rank: Some(1),
        ..request("legacy-rank-fallback", 0.0, 4, 2)
    }];

    let report = run_engine_replay(replay).unwrap();
    let routes = &report.per_request[0].routing_history;
    assert_eq!(
        routes
            .iter()
            .find(|route| route.pool == ReplayRequestPool::Prefill)
            .unwrap()
            .dp_rank,
        Some(1)
    );
    assert_eq!(
        routes
            .iter()
            .find(|route| route.pool == ReplayRequestPool::Decode)
            .unwrap()
            .dp_rank,
        Some(1)
    );
}

#[test]
fn aggregated_replay_rejects_prefill_dp_rank() {
    let mut replay = spec(engine_config(TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 1.0,
    }));
    replay.requests[0].prefill_dp_rank = Some(0);

    let error = run_engine_replay(replay).unwrap_err();
    assert!(error.to_string().contains("cannot specify prefill_dp_rank"));
}

#[test]
fn disaggregated_replay_validates_authored_rank_per_role() {
    for (prefill_dp_rank, decode_dp_rank, expected) in [
        (Some(2), Some(0), "prefill placement"),
        (Some(0), Some(4), "decode placement"),
    ] {
        let mut replay = disaggregated_spec(
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
        let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
        config.prefill.as_mut().unwrap().dp_size = 2;
        config.decode.as_mut().unwrap().dp_size = 4;
        replay.engine = serde_json::to_value(config).unwrap();
        replay.requests = vec![ReplayRequest {
            dp_rank: decode_dp_rank,
            prefill_dp_rank,
            ..request("out-of-range", 0.0, 4, 2)
        }];

        let error = run_engine_replay(replay).unwrap_err();
        assert!(error.to_string().contains(expected), "{error:#}");
    }
}

#[test]
fn sglang_disaggregated_attention_dp_uses_per_rank_prefill_chunks() {
    let run = |prefill_dp| {
        let mut replay = disaggregated_spec(
            Backend::Sglang,
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
        );
        let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine.clone()).unwrap();
        let prefill = config.prefill.as_mut().unwrap();
        prefill.dp_size = prefill_dp;
        prefill.rank.sglang.chunked_prefill_size = 16;
        replay.engine = serde_json::to_value(config).unwrap();
        replay.requests = vec![request("long-prompt", 0.0, 16, 1)];
        run_engine_replay(replay).unwrap()
    };

    let dp1 = run(1);
    let dp4 = run(4);
    let dp1_source_held_ms = dp1.per_request[0].source_held_ms.unwrap();
    let dp4_source_held_ms = dp4.per_request[0].source_held_ms.unwrap();
    assert!(
        dp4_source_held_ms >= dp1_source_held_ms + 3.0,
        "DP4 should execute four 4-token prefill chunks: dp1={dp1_source_held_ms}, dp4={dp4_source_held_ms}"
    );
}

// Independently authored scheduling sequences for the behavior documented at
// https://github.com/sgl-project/sglang/blob/20621aa14bda7726a8a968f326198eac61717fef/python/sglang/srt/managers/scheduler.py
// (prefill_decode_interval). No upstream implementation or tests are copied.
fn sglang_interval_config(interval: usize) -> EngineConfig {
    EngineConfig {
        num_gpu_blocks: 1024,
        block_size: 1,
        max_num_seqs: 16,
        max_num_batched_tokens: 128,
        prefill_decode_interval: interval,
        enable_prefix_caching: false,
        sglang: SglangConfig {
            chunked_prefill_size: 4,
            // A zero ratio cannot decay: an empty interval round must carry
            // its own retry signal to avoid the replay livelock guard.
            schedule_conservativeness: 0.0,
            ..SglangConfig::default()
        },
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 7.0,
            decode_ms: 2.0,
        },
        ..EngineConfig::for_backend(Backend::Sglang)
    }
}

fn sglang_interval_engine(mut config: EngineConfig, dp_size: u32) -> Engine {
    // The public chunk limit is divided among attention-DP ranks. Keep each
    // rank's four-token chunk constant across the scheduling scenarios.
    config.sglang.chunked_prefill_size *= dp_size as usize;
    EngineFactory::new(config)
        .unwrap()
        .build(EngineIdentity::new(0), NonZeroU32::new(dp_size).unwrap())
        .unwrap()
}

fn submit_interval_request(
    engine: &mut Engine,
    dp_rank: u32,
    id: u128,
    input_tokens: usize,
    output_tokens: usize,
    now_ms: f64,
) {
    engine
        .apply_command_effects(
            SchedulerCommand::new(
                dp_rank,
                Command::Submit(Request {
                    request_id: Uuid::from_u128(id),
                    tokens: vec![id as u32; input_tokens],
                    max_output_tokens: output_tokens,
                    output_token_ids: None,
                }),
            ),
            now_ms,
        )
        .unwrap();
}

struct IntervalRound {
    duration_ms: f64,
    retry: SameTimestampRetry,
    ranks: Vec<PassCompletionEffects>,
}

fn step_interval_engine(engine: &mut Engine, now_ms: &mut f64) -> IntervalRound {
    let started = engine.execute_pass(*now_ms).unwrap().unwrap();
    let completed = engine
        .complete_pass(started.pass_id, started.end_ms)
        .unwrap();
    *now_ms = started.end_ms;
    IntervalRound {
        duration_ms: started.end_ms - started.started_at_ms,
        retry: started.same_timestamp_retry,
        ranks: completed
            .effects
            .by_rank
            .into_iter()
            .enumerate()
            .map(|(rank, effects)| {
                assert_eq!(effects.dp_rank, rank as u32);
                effects.effects
            })
            .collect(),
    }
}

fn assert_interval_work(round: &IntervalRound, rank: usize, prefill: u32, decode: u32) {
    let fpm = &round.ranks[rank].forward_pass_metrics;
    assert_eq!(
        (fpm.num_prefill_requests, fpm.num_decode_requests),
        (prefill, decode),
        "rank {rank}: {fpm:?}"
    );
}

#[test]
fn sglang_split_prefixes_remain_reusable_across_cache_pressure() {
    let mut config = sglang_interval_config(0);
    config.block_size = 4;
    config.num_gpu_blocks = 64;
    config.enable_prefix_caching = true;
    config.sglang.chunked_prefill_size = 256;
    config.max_num_batched_tokens = 256;
    let mut engine = sglang_interval_engine(config, 1);
    let mut now_ms = 0.0;
    let seed: Vec<u32> = (0..64).collect();
    let mut cases = vec![(seed.clone(), Some(64))];
    for prefix_len in (4..64).step_by(4) {
        let mut tokens = seed[..prefix_len].to_vec();
        tokens.extend([1_000 + prefix_len as u32; 4]);
        cases.push((tokens, Some(4)));
    }
    // Force eviction after repeatedly splitting the original long edge.
    cases.push((vec![10_000; 252], Some(252)));
    cases.push((seed.clone(), None));
    let mut tokens = seed[..32].to_vec();
    tokens.extend([20_000; 4]);
    cases.push((tokens, Some(4)));

    for (index, (tokens, expected_prefill)) in cases.into_iter().enumerate() {
        let request_id = Uuid::from_u128(index as u128 + 1);
        engine
            .apply_command_effects(
                SchedulerCommand::new(
                    0,
                    Command::Submit(Request {
                        request_id,
                        tokens,
                        max_output_tokens: 0,
                        output_token_ids: None,
                    }),
                ),
                now_ms,
            )
            .unwrap();
        let round = step_interval_engine(&mut engine, &mut now_ms);
        if let Some(expected) = expected_prefill {
            assert_eq!(
                round.ranks[0].forward_pass_metrics.sum_prefill_tokens, expected,
                "request {index}"
            );
        }
        let outputs = &round.ranks[0].outputs;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, request_id);
        assert!(outputs[0].completed && !outputs[0].rejected);
        assert!(engine.is_drained());
    }
}

#[test]
fn sglang_prefill_packs_remaining_pages_and_completes_partial_chunks() {
    for (budget, cached_prefix, prompts, expected_work) in [
        (8, 0, vec![4, 8], vec![8, 4]),
        (6, 0, vec![5], vec![4, 1]),
        (8, 0, vec![6], vec![6]),
        (8, 0, vec![7, 8], vec![7, 8]),
        (8, 4, vec![4, 8], vec![8, 4]),
        (6, 4, vec![5], vec![4, 1]),
    ] {
        let mut config = sglang_interval_config(0);
        config.block_size = 4;
        config.sglang.chunked_prefill_size = budget;
        config.enable_prefix_caching = cached_prefix > 0;
        let mut engine = sglang_interval_engine(config, 1);
        let mut now_ms = 0.0;
        if cached_prefix > 0 {
            submit_interval_request(&mut engine, 0, 0, cached_prefix, 0, now_ms);
            let seed = step_interval_engine(&mut engine, &mut now_ms);
            assert!(seed.ranks[0].outputs[0].completed);
        }
        for (index, &prompt) in prompts.iter().enumerate() {
            let mut tokens = vec![0; cached_prefix];
            tokens.extend(std::iter::repeat_n(index as u32 + 1, prompt));
            engine
                .apply_command_effects(
                    SchedulerCommand::new(
                        0,
                        Command::Submit(Request {
                            request_id: Uuid::from_u128(index as u128 + 1),
                            tokens,
                            max_output_tokens: 0,
                            output_token_ids: None,
                        }),
                    ),
                    now_ms,
                )
                .unwrap();
        }

        let mut completed = Vec::new();
        for expected_tokens in expected_work {
            let round = step_interval_engine(&mut engine, &mut now_ms);
            assert_eq!(
                round.ranks[0].forward_pass_metrics.sum_prefill_tokens, expected_tokens,
                "budget={budget}, cached_prefix={cached_prefix}, prompts={prompts:?}"
            );
            for output in &round.ranks[0].outputs {
                assert!(output.completed && !output.rejected);
                assert_eq!(output.token_id, None);
                completed.push(output.request_id);
            }
        }
        completed.sort_unstable();
        assert_eq!(
            completed,
            (1..=prompts.len() as u128)
                .map(Uuid::from_u128)
                .collect::<Vec<_>>()
        );
        assert!(engine.is_drained());
    }
}

#[test]
fn sglang_interval_default_zero_preserves_consecutive_prefill_chunks() {
    assert_eq!(
        EngineConfig::for_backend(Backend::Sglang).prefill_decode_interval,
        0
    );
    let mut engine = sglang_interval_engine(sglang_interval_config(0), 1);
    let mut now_ms = 0.0;
    submit_interval_request(&mut engine, 0, 1, 12, 1, now_ms);

    for chunk in 0..3 {
        let round = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&round, 0, 1, 0);
        assert_eq!(round.ranks[0].forward_pass_metrics.sum_prefill_tokens, 4);
        assert_eq!(round.duration_ms, 7.0);
        assert_eq!(round.ranks[0].outputs.len(), usize::from(chunk == 2));
    }
    assert!(!engine.is_ready());
}

#[test]
fn sglang_interval_blocks_exactly_n_rounds_and_gives_running_decode_a_turn() {
    for interval in [0, 1, 2, 3] {
        let mut engine = sglang_interval_engine(sglang_interval_config(interval), 1);
        let mut now_ms = 0.0;
        submit_interval_request(&mut engine, 0, 1, 4, 100, now_ms);
        assert_interval_work(&step_interval_engine(&mut engine, &mut now_ms), 0, 1, 0);
        submit_interval_request(&mut engine, 0, 2, 4, 1, now_ms);

        for _ in 0..interval {
            let round = step_interval_engine(&mut engine, &mut now_ms);
            assert_interval_work(&round, 0, 0, 1);
            assert_eq!(round.duration_ms, 2.0);
            assert_eq!(round.ranks[0].outputs.len(), 1);
            assert_eq!(round.ranks[0].outputs[0].request_id, Uuid::from_u128(1));
            assert_eq!(round.ranks[0].metrics.waiting_requests, 1);
        }

        // Pure decode rounds did not arm the interval again. Admission resumes
        // immediately after the Nth blocked round, without a periodic phase.
        let resumed = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&resumed, 0, 1, 0);
        assert_eq!(resumed.ranks[0].outputs[0].request_id, Uuid::from_u128(2));
        assert_eq!(now_ms, 14.0 + 2.0 * interval as f64);
    }
}

#[test]
fn sglang_interval_gates_every_long_prompt_chunk_without_running_decode() {
    let mut engine = sglang_interval_engine(sglang_interval_config(2), 1);
    let mut now_ms = 0.0;
    submit_interval_request(&mut engine, 0, 1, 12, 1, now_ms);

    for chunk in 0..3 {
        let round = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&round, 0, 1, 0);
        assert_eq!(round.ranks[0].forward_pass_metrics.sum_prefill_tokens, 4);
        assert_eq!(round.ranks[0].outputs.len(), usize::from(chunk == 2));
        if chunk == 2 {
            break;
        }
        for remaining in [1, 0] {
            let blocked = step_interval_engine(&mut engine, &mut now_ms);
            assert_interval_work(&blocked, 0, 0, 0);
            assert_eq!(blocked.duration_ms, 0.0);
            assert_eq!(blocked.retry, SameTimestampRetry::Countdown { remaining });
            assert!(blocked.ranks[0].outputs.is_empty());
        }
    }
    // Scheduler-only rounds contribute no fabricated GPU latency.
    assert_eq!(now_ms, 21.0);
}

#[test]
fn sglang_interval_counts_speculative_rounds_instead_of_output_tokens() {
    let mut config = sglang_interval_config(2);
    config.aic_nextn = Some(2);
    config.aic_nextn_accept_rates = Some("1,1".to_string());
    let mut engine = sglang_interval_engine(config, 1);
    let mut now_ms = 0.0;
    submit_interval_request(&mut engine, 0, 1, 4, 100, now_ms);
    step_interval_engine(&mut engine, &mut now_ms);
    submit_interval_request(&mut engine, 0, 2, 4, 1, now_ms);

    for _ in 0..2 {
        let round = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&round, 0, 0, 1);
        assert_eq!(round.ranks[0].outputs.len(), 3);
        assert!(
            round.ranks[0]
                .outputs
                .iter()
                .all(|output| output.request_id == Uuid::from_u128(1))
        );
    }
    let resumed = step_interval_engine(&mut engine, &mut now_ms);
    assert_interval_work(&resumed, 0, 1, 0);
    assert_eq!(resumed.ranks[0].outputs[0].request_id, Uuid::from_u128(2));
}

#[test]
fn sglang_interval_globally_rearms_a_rank_that_only_decoded() {
    let mut engine = sglang_interval_engine(sglang_interval_config(2), 2);
    let mut now_ms = 0.0;
    submit_interval_request(&mut engine, 0, 1, 4, 100, now_ms);
    step_interval_engine(&mut engine, &mut now_ms);
    for _ in 0..2 {
        step_interval_engine(&mut engine, &mut now_ms);
    }

    submit_interval_request(&mut engine, 1, 2, 4, 100, now_ms);
    let extended = step_interval_engine(&mut engine, &mut now_ms);
    assert_interval_work(&extended, 0, 0, 1);
    assert_interval_work(&extended, 1, 1, 0);
    submit_interval_request(&mut engine, 0, 3, 4, 1, now_ms);
    for _ in 0..2 {
        let blocked = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&blocked, 0, 0, 1);
        assert_interval_work(&blocked, 1, 0, 1);
        assert_eq!(blocked.ranks[0].metrics.waiting_requests, 1);
    }
    let resumed = step_interval_engine(&mut engine, &mut now_ms);
    assert_interval_work(&resumed, 0, 1, 0);
    assert_interval_work(&resumed, 1, 0, 1);
    assert_eq!(resumed.ranks[0].outputs[0].request_id, Uuid::from_u128(3));
}

#[test]
fn sglang_interval_arms_idle_dp_sibling_and_counts_its_idle_rounds() {
    // Exercise both positions so rank iteration order cannot hide a local arm.
    for prefill_rank in [0, 1] {
        let idle_rank = 1 - prefill_rank;
        let mut engine = sglang_interval_engine(sglang_interval_config(2), 2);
        let mut now_ms = 0.0;
        submit_interval_request(&mut engine, prefill_rank, 1, 4, 100, now_ms);
        let extended = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&extended, prefill_rank as usize, 1, 0);
        assert_interval_work(&extended, idle_rank as usize, 0, 0);

        // One global scheduler round elapses while the sibling is still idle.
        let first_blocked = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&first_blocked, idle_rank as usize, 0, 0);
        submit_interval_request(&mut engine, idle_rank, 2, 4, 1, now_ms);
        let second_blocked = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&second_blocked, idle_rank as usize, 0, 0);
        assert_interval_work(&second_blocked, prefill_rank as usize, 0, 1);
        let resumed = step_interval_engine(&mut engine, &mut now_ms);
        assert_interval_work(&resumed, idle_rank as usize, 1, 0);
        assert_eq!(
            resumed.ranks[idle_rank as usize].outputs[0].request_id,
            Uuid::from_u128(2)
        );
    }
}

#[test]
fn sglang_interval_drains_fully_idle_rounds_before_a_later_request_wave() {
    let mut engine = sglang_interval_engine(sglang_interval_config(2), 2);
    let mut now_ms = 0.0;
    submit_interval_request(&mut engine, 0, 1, 4, 1, now_ms);
    let prefill = step_interval_engine(&mut engine, &mut now_ms);
    assert!(prefill.ranks[0].outputs[0].completed);

    for _ in 0..2 {
        assert!(engine.is_ready());
        let idle = step_interval_engine(&mut engine, &mut now_ms);
        assert_eq!(idle.duration_ms, 0.0);
        assert_interval_work(&idle, 0, 0, 0);
        assert_interval_work(&idle, 1, 0, 0);
    }
    assert!(!engine.is_ready());
    assert_eq!(now_ms, 7.0);

    now_ms = 50.0;
    submit_interval_request(&mut engine, 1, 2, 4, 1, now_ms);
    let next_wave = step_interval_engine(&mut engine, &mut now_ms);
    assert_interval_work(&next_wave, 1, 1, 0);
    assert_eq!(now_ms, 57.0);
}

#[test]
fn sglang_interval_zero_duration_chunk_replay_makes_deterministic_progress() {
    for dp_size in [1, 2] {
        // The large interval exceeds replay's generic zero-time retry limit.
        // Its finite countdown still must complete without weakening that guard.
        for interval in [3, 1025] {
            let mut rank = sglang_interval_config(interval);
            rank.sglang.chunked_prefill_size *= dp_size as usize;
            rank.timing_model = TimingModelConfig::Fixed {
                prefill_ms: 0.0,
                decode_ms: 0.0,
            };
            let mut replay = spec(ReplayEngineConfig {
                dp_size,
                rank,
                ..ReplayEngineConfig::default()
            });
            replay.requests = vec![ReplayRequest {
                dp_rank: Some(0),
                ..request("only-chunked-prefill", 0.0, 12, 1)
            }];
            let first = run_canonical_engine_replay(replay.clone());
            let second = run_canonical_engine_replay(replay);
            assert_deterministic(&first, &second);
            assert_eq!(first.request_counts.completed_requests, 1);
            assert_eq!(first.throughput.duration_ms, 0.0);
            assert_eq!(first.per_request[0].first_token_ms, Some(0.0));
        }
    }
}

#[test]
fn sglang_interval_does_not_hide_an_impossible_request_after_countdown() {
    let mut config = impossible_sglang_config();
    config.rank.num_gpu_blocks = 2;
    config.rank.prefill_decode_interval = 1025;
    config.rank.sglang.schedule_conservativeness = 0.0;
    let mut replay = spec(config);
    replay.requests = vec![
        request("fits-and-arms-interval", 0.0, 4, 1),
        request("impossible", 0.0, 12, 2),
    ];

    let error = run_engine_replay(replay).unwrap_err();
    assert_eq!(
        error.to_string(),
        "replay invariant violated: offline replay detected an effect-free zero-duration pass with 1 in-flight requests remaining"
    );
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
    assert_eq!(first.throughput.duration_ms, 12.0);
    assert_eq!(first.throughput.decode_gpus_per_worker, 1);
    assert_eq!(first.per_request[0].first_token_ms, Some(10.0));
    assert_eq!(first.per_request[0].terminal_time_ms, 12.0);
}

#[derive(Debug, PartialEq, Eq)]
enum TimingCall {
    Prefill(usize, usize, usize),
    Decode(usize, usize, usize),
}

#[derive(Default)]
struct RecordingTiming {
    calls: Mutex<Vec<TimingCall>>,
}

impl TimingModel for RecordingTiming {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        self.calls
            .lock()
            .unwrap()
            .push(TimingCall::Prefill(batch_size, mean_isl, mean_prefix));
        Ok(10.0)
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        self.calls.lock().unwrap().push(TimingCall::Decode(
            batch_size,
            active_kv_tokens,
            mean_context_length,
        ));
        Ok(2.0)
    }
}

fn recording_timing_spec() -> ReplaySpec {
    spec(engine_config(TimingModelConfig::External {
        provider: "recording".to_string(),
        config: serde_json::Value::Null,
    }))
}

#[test]
fn vllm_first_token_uses_the_final_prefill_forward_once() {
    for output_tokens in [0, 1, 2] {
        let timing = Arc::new(RecordingTiming::default());
        let mut replay = recording_timing_spec();
        replay.requests = vec![request("request", 0.0, 4, output_tokens)];
        let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
        let record = &report.per_request[0];
        assert_eq!(record.output_length, output_tokens);
        assert_eq!(record.first_token_ms, (output_tokens > 0).then_some(10.0));
        assert_eq!(
            record.terminal_time_ms,
            if output_tokens == 2 { 12.0 } else { 10.0 }
        );
        let mut expected = vec![TimingCall::Prefill(1, 4, 0)];
        if output_tokens == 2 {
            expected.push(TimingCall::Decode(1, 5, 5));
        }
        assert_eq!(*timing.calls.lock().unwrap(), expected);
    }
}

#[test]
fn vllm_prefill_batch_samples_all_first_tokens_without_a_decode_query() {
    let timing = Arc::new(RecordingTiming::default());
    let mut replay = recording_timing_spec();
    replay.requests = vec![
        request_with_tokens("first", 0.0, vec![1, 2, 3, 4], 1),
        request_with_tokens("second", 0.0, vec![5, 6, 7, 8], 1),
    ];
    let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
    assert_eq!(report.request_counts.completed_requests, 2);
    assert!(report.per_request.iter().all(|row| {
        row.output_length == 1 && row.first_token_ms == Some(10.0) && row.terminal_time_ms == 10.0
    }));
    assert_eq!(
        *timing.calls.lock().unwrap(),
        vec![TimingCall::Prefill(2, 4, 0)]
    );
}

#[test]
fn first_token_timing_fix_preserves_trtllm_and_speculative_paths() {
    for backend in [Backend::Trtllm, Backend::Vllm] {
        let timing = Arc::new(RecordingTiming::default());
        let mut config = engine_config(TimingModelConfig::External {
            provider: "recording".to_string(),
            config: serde_json::Value::Null,
        });
        config.rank.backend = backend;
        if backend == Backend::Vllm {
            config.rank.aic_nextn = Some(1);
            config.rank.aic_nextn_accept_rates = Some("1.0".to_string());
        }
        let mut replay = spec(config);
        replay.requests = vec![request("request", 0.0, 4, 1)];
        let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
        assert_eq!(report.per_request[0].first_token_ms, Some(12.0));
        assert_eq!(report.per_request[0].terminal_time_ms, 12.0);
        assert_eq!(
            *timing.calls.lock().unwrap(),
            vec![TimingCall::Prefill(1, 4, 0), TimingCall::Decode(1, 4, 4),]
        );
    }
}

#[test]
fn vllm_mixed_pass_prices_only_ongoing_decoders_and_completes_together() {
    let timing = Arc::new(RecordingTiming::default());
    let mut replay = recording_timing_spec();
    replay.requests = vec![
        request_with_tokens("decoding", 0.0, vec![1, 2, 3, 4], 2),
        request_with_tokens("prefilling", 5.0, (10..18).collect(), 1),
    ];
    let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
    let decoding = report
        .per_request
        .iter()
        .find(|row| row.request_id.as_deref() == Some("decoding"))
        .unwrap();
    let prefilling = report
        .per_request
        .iter()
        .find(|row| row.request_id.as_deref() == Some("prefilling"))
        .unwrap();
    assert_eq!(decoding.first_token_ms, Some(10.0));
    assert_eq!(decoding.terminal_time_ms, 22.0);
    assert_eq!(prefilling.first_token_ms, Some(22.0));
    assert_eq!(prefilling.terminal_time_ms, 22.0);
    assert_eq!(
        *timing.calls.lock().unwrap(),
        vec![
            TimingCall::Prefill(1, 4, 0),
            TimingCall::Prefill(1, 8, 0),
            TimingCall::Decode(1, 5, 5),
        ]
    );
}

#[test]
fn vllm_chunked_prefill_waits_for_the_final_chunk_before_sampling() {
    let timing = Arc::new(RecordingTiming::default());
    let mut replay = recording_timing_spec();
    let mut config: ReplayEngineConfig = serde_json::from_value(replay.engine).unwrap();
    config.rank.max_num_batched_tokens = 4;
    replay.engine = serde_json::to_value(config).unwrap();
    replay.requests = vec![request("chunked", 0.0, 10, 2)];
    let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
    assert_eq!(report.per_request[0].first_token_ms, Some(30.0));
    assert_eq!(report.per_request[0].terminal_time_ms, 32.0);
    assert_eq!(
        *timing.calls.lock().unwrap(),
        vec![
            TimingCall::Prefill(1, 4, 0),
            TimingCall::Prefill(1, 8, 4),
            TimingCall::Prefill(1, 10, 8),
            TimingCall::Decode(1, 11, 11),
        ]
    );
}

#[test]
fn vllm_cached_prefill_recomputes_logits_without_an_extra_decode_forward() {
    for prompt in [
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![1, 2, 3, 4, 9, 10, 11, 12],
    ] {
        let timing = Arc::new(RecordingTiming::default());
        let mut replay = recording_timing_spec();
        replay.requests = vec![
            request_with_tokens("seed", 0.0, (1..9).collect(), 0),
            request_with_tokens("reuse", 20.0, prompt, 1),
        ];
        let report = run_engine_replay_with_timing(replay, timing.clone()).unwrap();
        let reuse = report
            .per_request
            .iter()
            .find(|row| row.request_id.as_deref() == Some("reuse"))
            .unwrap();
        assert_eq!(reuse.reused_input_tokens, 4);
        assert_eq!(reuse.first_token_ms, Some(30.0));
        assert_eq!(reuse.terminal_time_ms, 30.0);
        assert_eq!(
            *timing.calls.lock().unwrap(),
            vec![TimingCall::Prefill(1, 8, 0), TimingCall::Prefill(1, 8, 4),]
        );
    }
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
fn native_host_offload_rejects_attention_dp_and_disaggregated_roles() {
    let mut config = engine_config(TimingModelConfig::Fixed {
        prefill_ms: 0.0,
        decode_ms: 0.0,
    });
    config.rank.kv_cache_bytes_per_token = Some(1);
    config.rank.native_host_offload =
        Some(NativeHostOffloadConfig::new(1).with_bandwidths(1.0, 1.0));

    let mut attention_dp = config.clone();
    attention_dp.dp_size = 2;
    let error = run_engine_replay(spec(attention_dp)).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("dp_size=1"), "{message}");

    let mut disaggregated = spec(config);
    disaggregated.topology = ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 0.0,
    };
    let error = run_engine_replay(disaggregated).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("only aggregated replay"), "{message}");
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
    assert_eq!(report.throughput.duration_ms, 6.0);
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
fn native_trtllm_disaggregated_replay_completes() {
    let spec = disaggregated_spec(
        Backend::Trtllm,
        TimingModelConfig::Polynomial,
        TimingModelConfig::Polynomial,
    );
    let report = run_engine_replay(spec).unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.request_counts.total_input_tokens, 4);
    assert_eq!(report.request_counts.total_output_tokens, 2);
}
