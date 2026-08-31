// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{
    Backend, EngineConfig, KvEvent, NativeHostOffloadConfig, TimingModelConfig,
};
use aisimulate_core::replay::loadgen::{SessionTrace, Trace, TurnTrace, WorkloadDriver};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ProviderSpec, ReplayAdapters, ReplayArtifactHostOffloadEvent,
    ReplayArtifactHostOffloadEventData, ReplayArtifactKvEventVisibility, ReplayArtifacts,
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayReport,
    ReplayRequest, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer, RoundRobinComposition,
    WorkerPoolSpec,
};

fn spec(backend: Backend, workers: usize, dp_size: u32) -> ReplaySpec {
    let engine = ReplayEngineConfig {
        dp_size,
        rank: EngineConfig {
            num_gpu_blocks: 64,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 2.0,
            },
            ..EngineConfig::for_backend(backend)
        },
        ..ReplayEngineConfig::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms: 0.0,
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
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

fn turn(index: usize) -> TurnTrace {
    let first_hash = 11 + u32::try_from(index).unwrap() * 10;
    TurnTrace {
        input_length: 8,
        max_output_tokens: 2,
        hash_ids: vec![first_hash, first_hash + 1],
        ..TurnTrace::default()
    }
}

fn workload(arrivals_ms: &[f64]) -> WorkloadDriver {
    Trace {
        block_size: 4,
        sessions: arrivals_ms
            .iter()
            .enumerate()
            .map(|(index, arrival_ms)| SessionTrace {
                session_id: format!("session-{index}"),
                first_arrival_timestamp_ms: Some(*arrival_ms),
                turns: vec![turn(index)],
            })
            .collect(),
    }
    .into_trace_driver_with_block_size(4)
    .unwrap()
}

fn replayer(spec: ReplaySpec, arrivals_ms: &[f64]) -> Replayer<RoundRobinComposition> {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(workload(arrivals_ms)))
        .with_capture_options(ReplayCaptureOptions {
            determinism: ReplayDeterminism::CanonicalV1,
            ..ReplayCaptureOptions::default()
        })
}

fn host_offload_spec(capacity_blocks: usize) -> ReplaySpec {
    let mut replay_spec = spec(Backend::Vllm, 1, 1);
    let mut engine: ReplayEngineConfig =
        serde_json::from_value(replay_spec.engine.clone()).unwrap();
    engine.rank.num_gpu_blocks = 1;
    engine.rank.timing_model = TimingModelConfig::Fixed {
        prefill_ms: 0.0,
        decode_ms: 0.0,
    };
    engine.rank.kv_cache_bytes_per_token = Some(250_000);
    engine.rank.native_host_offload =
        Some(NativeHostOffloadConfig::new(capacity_blocks).with_bandwidths(1.0, 1.0));
    replay_spec.engine = serde_json::to_value(engine).unwrap();
    replay_spec
}

fn replay_request(id: &str, arrival_time_ms: f64, input_token_ids: Vec<u32>) -> ReplayRequest {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "arrival_time_ms": arrival_time_ms,
        "input_tokens": input_token_ids.len(),
        "input_token_ids": input_token_ids,
        "output_tokens": 0,
    }))
    .unwrap()
}

fn assert_host_event(
    event: &ReplayArtifactHostOffloadEvent,
    request_id: uuid::Uuid,
    observed_at_ms: f64,
) {
    assert_eq!(event.request_id, request_id);
    assert_eq!(event.observed_at_ms, observed_at_ms);
}

fn capture(
    backend: Backend,
    visibility: ReplayArtifactKvEventVisibility,
) -> (ReplayReport, ReplayArtifacts) {
    replayer(spec(backend, 1, 1), &[0.0])
        .run_with_artifacts(visibility)
        .unwrap()
}

fn kv_parts(artifacts: &ReplayArtifacts) -> (Vec<KvEvent>, Vec<f64>) {
    artifacts
        .kv_events
        .iter()
        .map(|event| (event.event.clone(), event.observed_at_ms))
        .unzip()
}

#[test]
fn common_agg_runtime_captures_requests_outputs_and_the_same_report() {
    let replay_spec = spec(Backend::Vllm, 1, 1);
    let report = replayer(replay_spec.clone(), &[0.0]).run().unwrap();
    let (artifact_report, artifacts) = replayer(replay_spec, &[0.0])
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    assert_eq!(
        serde_json::to_value(report.clone().with_wall_time_ms(0.0)).unwrap(),
        serde_json::to_value(artifact_report.clone().with_wall_time_ms(0.0)).unwrap()
    );
    assert_eq!(
        serde_json::to_value(report.per_request).unwrap(),
        serde_json::to_value(artifact_report.per_request).unwrap()
    );

    let request = &artifacts.requests[0];
    assert_eq!(
        (
            request.observed_at_ms,
            request.scheduled_ready_at_ms,
            request.input_length,
            request.output_length,
        ),
        (0.0, 0.0, 8, 2)
    );
    assert_eq!(
        request.replay_hashes,
        Some(turn(0).to_replay_hashes(4, 4).unwrap())
    );
    assert_eq!(
        artifacts
            .outputs
            .iter()
            .map(|output| {
                (
                    output.observed_at_ms,
                    output.request_id == request.request_id,
                    output.token_id.is_some(),
                    output.completed,
                    output.rejected,
                    output.cached_tokens,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (12.0, true, true, false, false, Some(0)),
            (14.0, true, true, true, false, None),
        ]
    );
    assert!(artifacts.host_offload_events.is_empty());
}

#[test]
fn request_arrivals_and_hashes_use_the_agg_event_loop_clock() {
    let (_, artifacts) = replayer(spec(Backend::Vllm, 1, 1), &[0.0, 5.0])
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    assert_eq!(
        artifacts
            .requests
            .iter()
            .map(|request| (request.observed_at_ms, request.scheduled_ready_at_ms))
            .collect::<Vec<_>>(),
        vec![(0.0, 0.0), (5.0, 5.0)]
    );
}

#[test]
fn native_and_normalized_kv_visibility_preserve_raw_order() {
    for backend in [Backend::Vllm, Backend::Trtllm, Backend::Sglang] {
        let (_, native) = capture(backend, ReplayArtifactKvEventVisibility::Native);
        let (_, start) = capture(backend, ReplayArtifactKvEventVisibility::PassStart);
        let (_, end) = capture(backend, ReplayArtifactKvEventVisibility::PassEnd);
        let (events, native_times) = kv_parts(&native);
        let (start_events, start_times) = kv_parts(&start);
        let (end_events, end_times) = kv_parts(&end);
        assert!(!events.is_empty());
        assert_eq!(events, start_events);
        assert_eq!(events, end_events);
        assert!(start_times.iter().zip(&end_times).all(|(a, b)| a <= b));
        assert!(start_times.iter().zip(&end_times).any(|(a, b)| a < b));
        assert_eq!(native_times, end_times);
    }
}

#[test]
fn capped_passes_respect_visibility_boundaries() {
    let capped = |visibility| {
        let mut replay_spec = spec(Backend::Vllm, 1, 1);
        replay_spec.max_sim_time_ms = Some(1.0);
        replayer(replay_spec, &[0.0])
            .run_with_artifacts(visibility)
            .unwrap()
            .1
    };
    let native = capped(ReplayArtifactKvEventVisibility::Native);
    let start = capped(ReplayArtifactKvEventVisibility::PassStart);
    let end = capped(ReplayArtifactKvEventVisibility::PassEnd);
    assert!(native.kv_events.is_empty());
    assert!(start.kv_events.is_empty());
    assert!(end.kv_events.is_empty());
}

#[test]
fn artifact_capture_rejects_unsupported_topologies() {
    let error = |replay_spec| {
        replayer(replay_spec, &[0.0])
            .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
            .unwrap_err()
            .to_string()
    };
    for replay_spec in [spec(Backend::Vllm, 2, 1), spec(Backend::Vllm, 1, 2)] {
        assert!(error(replay_spec).contains("one logical DP1 worker"));
    }
    let mut disagg = spec(Backend::Vllm, 1, 1);
    disagg.topology = ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 0.0,
    };
    assert!(error(disagg).contains("require aggregated topology"));
}

#[test]
fn h2d_activation_is_captured_at_the_internal_work_boundary() {
    let mut replay_spec = host_offload_spec(2);
    replay_spec.requests = [
        ("seed", 0.0, vec![1, 2, 3, 4]),
        ("second", 5.0, vec![5, 6, 7, 8]),
        ("evict", 10.0, vec![9, 10, 11, 12]),
        ("restore", 15.0, vec![5, 6, 7, 8]),
    ]
    .into_iter()
    .map(|(id, arrival_time_ms, tokens)| replay_request(id, arrival_time_ms, tokens))
    .collect();

    let (_, artifacts) = Replayer::new(replay_spec, ReplayEngineFactory::new())
        .unwrap()
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    assert!(
        artifacts
            .kv_events
            .iter()
            .any(|event| event.observed_at_ms == 16.0)
    );

    let ids: Vec<_> = artifacts
        .requests
        .iter()
        .map(|request| request.request_id)
        .collect();
    let store_times = [(0.0, 1.0), (5.0, 6.0), (10.0, 11.0)];
    let (mut prepared, mut submitted, mut completed) = (0, 0, 0);
    let (mut queued, mut loaded, mut evicted) = (0, 0, 0);
    let mut stored_hashes = Vec::new();
    for event in &artifacts.host_offload_events {
        match &event.event {
            ReplayArtifactHostOffloadEventData::StorePrepared { .. } => {
                assert_host_event(event, ids[prepared], store_times[prepared].0);
                prepared += 1;
            }
            ReplayArtifactHostOffloadEventData::StoreSubmitted {
                completes_at_ms, ..
            } => {
                assert_host_event(event, ids[submitted], store_times[submitted].0);
                assert_eq!(*completes_at_ms, store_times[submitted].1);
                submitted += 1;
            }
            ReplayArtifactHostOffloadEventData::StoreCompleted { block_hashes, .. } => {
                assert_host_event(event, ids[completed], store_times[completed].1);
                assert_eq!(block_hashes.len(), 1);
                stored_hashes.push(block_hashes[0]);
                completed += 1;
            }
            ReplayArtifactHostOffloadEventData::LoadQueued {
                completes_at_ms, ..
            } => {
                assert_host_event(event, ids[3], 15.0);
                assert_eq!(*completes_at_ms, 16.0);
                queued += 1;
            }
            ReplayArtifactHostOffloadEventData::LoadCompleted { block_hashes, .. } => {
                assert_host_event(event, ids[3], 16.0);
                assert_eq!(block_hashes.as_slice(), &stored_hashes[1..2]);
                loaded += 1;
            }
            ReplayArtifactHostOffloadEventData::Evicted { block_hash } => {
                assert_host_event(event, ids[2], 10.0);
                assert_eq!(*block_hash, stored_hashes[0]);
                evicted += 1;
            }
            _ => {}
        }
    }
    assert_eq!(
        (prepared, submitted, completed, queued, loaded, evicted),
        (3, 3, 3, 1, 1, 1)
    );
}

#[test]
fn prepared_store_artifact_maps_hashes_to_request_local_block_indices() {
    let mut replay_spec = host_offload_spec(2);
    let mut engine: ReplayEngineConfig =
        serde_json::from_value(replay_spec.engine.clone()).unwrap();
    engine.rank.num_gpu_blocks = 2;
    replay_spec.engine = serde_json::to_value(engine).unwrap();
    replay_spec.requests = vec![
        replay_request("seed", 0.0, vec![1, 2, 3, 4]),
        replay_request("shared-prefix-suffix", 5.0, vec![1, 2, 3, 4, 5, 6, 7, 8]),
    ];

    let (_, artifacts) = Replayer::new(replay_spec, ReplayEngineFactory::new())
        .unwrap()
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    let mappings = artifacts
        .host_offload_events
        .windows(2)
        .filter_map(|events| {
            let [prepared, mapped] = events else {
                unreachable!()
            };
            let (
                ReplayArtifactHostOffloadEventData::StorePrepared {
                    transfer_id,
                    block_hashes,
                },
                ReplayArtifactHostOffloadEventData::StoreBlockMappings {
                    transfer_id: mapped_transfer_id,
                    mappings,
                },
            ) = (&prepared.event, &mapped.event)
            else {
                return None;
            };
            assert_eq!(prepared.request_id, mapped.request_id);
            assert_eq!(transfer_id, mapped_transfer_id);
            assert_eq!(mappings.len(), 1);
            assert_eq!(block_hashes.as_slice(), [mappings[0].block_hash]);
            Some((mapped.request_id, mappings[0].logical_block_index))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        mappings,
        vec![
            (artifacts.requests[0].request_id, 0),
            (artifacts.requests[1].request_id, 1),
        ]
    );
}
