// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, KvEvictionPolicy, TimingModelConfig};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayError, ReplayReport, ReplayRequest, ReplaySpec, ReplayTopology,
    Replayer, WorkerPoolSpec,
};
use serde_json::{Value, json};

const BACKENDS: [Backend; 3] = [Backend::Vllm, Backend::Sglang, Backend::Trtllm];

fn request(index: usize, tokens: Vec<u32>, output_tokens: usize) -> ReplayRequest {
    ReplayRequest {
        id: format!("request-{index}"),
        arrival_time_ms: index as f64 * 100.0,
        input_tokens: tokens.len(),
        input_token_ids: Some(tokens),
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        prefill_dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: Value::Null,
    }
}

fn spec(backend: Backend, workers: usize, requests: Vec<ReplayRequest>) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: ReplayTopology::aggregated(workers),
        engine: serde_json::to_value(ReplayEngineConfig {
            rank: EngineConfig {
                block_size: 4,
                num_gpu_blocks: 64,
                max_num_seqs: 4,
                max_num_batched_tokens: 64,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                },
                ..EngineConfig::for_backend(backend)
            },
            ..Default::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests,
    }
}

fn run(spec: ReplaySpec) -> Result<ReplayReport, ReplayError> {
    Replayer::new(spec, ReplayEngineFactory::new())?
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run()
}

fn belady(mut spec: ReplaySpec) -> ReplaySpec {
    spec.engine["kv_eviction_policy"] = json!("belady");
    spec
}

fn stable_summary(report: &ReplayReport) -> Value {
    let mut summary = serde_json::to_value(report).unwrap();
    let fields = summary.as_object_mut().unwrap();
    for key in [
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    ] {
        fields.remove(key);
    }
    summary
}

#[test]
fn omitted_eviction_policy_preserves_explicit_lru_results() {
    for backend in BACKENDS {
        let mut input = spec(
            backend,
            1,
            vec![request(0, vec![1; 16], 2), request(1, vec![1; 16], 2)],
        );
        input
            .engine
            .as_object_mut()
            .unwrap()
            .remove("kv_eviction_policy");
        let implicit = run(input.clone()).unwrap();
        input.engine["kv_eviction_policy"] = json!("lru");
        let explicit = run(input).unwrap();
        assert_eq!(implicit.kv_eviction_policy, KvEvictionPolicy::Lru);
        assert_eq!(implicit.kv_eviction_assumption, None);
        assert_eq!(stable_summary(&implicit), stable_summary(&explicit));
        assert_eq!(
            serde_json::to_value(&implicit.per_request).unwrap(),
            serde_json::to_value(&explicit.per_request).unwrap(),
        );
    }
}

#[test]
fn future_input_demand_does_not_create_kv_or_release_requests_early() {
    for backend in BACKENDS {
        let report = run(belady(spec(
            backend,
            1,
            vec![request(0, vec![7; 16], 1), request(1, vec![7; 16], 1)],
        )))
        .unwrap();
        assert_eq!(report.request_counts.completed_requests, 2, "{backend:?}");
        let first = &report.per_request[0];
        let second = &report.per_request[1];
        assert_eq!(first.admission_history[0].reused_input_tokens, 0);
        assert_eq!(second.arrival_time_ms, 100.0);
        assert_eq!(second.first_admit_ms, Some(100.0));
        assert!(second.admission_history[0].reused_input_tokens > 0);
        assert_eq!(report.kv_eviction_policy, KvEvictionPolicy::Belady);
        let summary = stable_summary(&report);
        assert_eq!(summary["kv_eviction_policy"], "belady");
        assert!(summary["kv_eviction_assumption"].is_string());
        assert_eq!(
            summary["committed_prefill_tokens"],
            report.committed_prefill_tokens
        );
    }
}

#[test]
fn committed_prefill_counts_chunks_once_and_excludes_decode_work() {
    for backend in BACKENDS {
        for use_belady in [false, true] {
            let mut input = spec(
                backend,
                1,
                vec![request(0, vec![1; 13], 3), request(1, vec![2; 21], 5)],
            );
            input.engine["rank"]["max_num_batched_tokens"] = json!(8);
            input.engine["rank"]["sglang"]["chunked_prefill_size"] = json!(8);
            let report = run(if use_belady { belady(input) } else { input }).unwrap();
            assert_eq!(report.request_counts.completed_requests, 2);
            assert_eq!(report.request_counts.total_output_tokens, 8);
            assert_eq!(
                report.committed_prefill_tokens, 34,
                "{backend:?}, {use_belady}"
            );
        }
    }
}

#[test]
fn committed_prefill_excludes_an_unfinished_pass_at_the_cutoff() {
    for backend in BACKENDS {
        let mut input = belady(spec(backend, 1, vec![request(0, vec![1; 16], 2)]));
        assert_eq!(run(input.clone()).unwrap().committed_prefill_tokens, 16);
        input.max_sim_time_ms = Some(0.5);
        let report = run(input).unwrap();
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(report.committed_prefill_tokens, 0, "{backend:?}");
        assert!(report.throughput.duration_ms <= 0.5);
    }
}

#[test]
fn fixed_multiple_workers_share_forecast_but_keep_caches_local() {
    for backend in BACKENDS {
        for workers in [2, 4] {
            let requests = (0..workers * 2)
                .map(|index| request(index, vec![7; 16], 1))
                .collect();
            let report = run(belady(spec(backend, workers, requests))).unwrap();
            assert_eq!(report.request_counts.completed_requests, workers * 2);
            let mut first_per_worker = std::collections::BTreeSet::new();
            for record in &report.per_request {
                let worker = record.decode_worker_idx.unwrap();
                if first_per_worker.insert(worker) {
                    // Global demand does not create remotely resident KV or
                    // predict a future route. Each worker starts genuinely cold.
                    assert_eq!(record.admission_history[0].reused_input_tokens, 0);
                } else {
                    assert!(record.admission_history[0].reused_input_tokens > 0);
                }
            }
            assert_eq!(first_per_worker.len(), workers);
        }
    }
}

#[test]
fn best_effort_belady_preserves_paused_prefix_through_cold_churn() {
    for backend in BACKENDS {
        // A pauses while three single-use prompts arrive, then resumes. Zero
        // output isolates page demand; fixed timing is not a throughput model.
        let requests = [1, 2, 3, 4, 1]
            .into_iter()
            .enumerate()
            .map(|(index, token)| request(index, vec![token; 4], 0))
            .collect();
        let mut input = spec(backend, 1, requests);
        input.engine["rank"]["num_gpu_blocks"] = json!(3);
        let lru = run(input.clone()).unwrap();
        let forecast = run(belady(input)).unwrap();
        assert_eq!(lru.request_counts.completed_requests, 5);
        assert_eq!(forecast.request_counts.completed_requests, 5);
        assert!(
            forecast.committed_prefill_tokens < lru.committed_prefill_tokens,
            "{backend:?}: Belady={} LRU={}",
            forecast.committed_prefill_tokens,
            lru.committed_prefill_tokens,
        );
        assert!(
            forecast.first_admission_prefix_cache_reused_ratio
                > lru.first_admission_prefix_cache_reused_ratio
        );
    }
}

#[test]
fn local_growing_prefixes_remain_competitive_under_lru() {
    for backend in BACKENDS {
        let requests = (0..6)
            .map(|index| request(index, vec![7; 4 * (index + 1)], 1))
            .collect();
        let input = spec(backend, 1, requests);
        let lru = run(input.clone()).unwrap();
        let forecast = run(belady(input)).unwrap();
        assert_eq!(forecast.request_counts.completed_requests, 6);
        assert_eq!(
            forecast.committed_prefill_tokens,
            lru.committed_prefill_tokens
        );
        assert_eq!(
            forecast.first_admission_prefix_cache_reused_ratio,
            lru.first_admission_prefix_cache_reused_ratio,
        );
    }
}

#[test]
fn input_only_forecast_preserves_native_reuse_of_generated_output() {
    for backend in BACKENDS {
        for use_belady in [false, true] {
            let mut first = request(0, vec![7; 4], 5);
            first.output_token_ids = Some(vec![9; 5]);
            let second = request(1, [vec![7; 4], vec![9; 4]].concat(), 0);
            let input = spec(backend, 1, vec![first, second]);
            let report = run(if use_belady { belady(input) } else { input }).unwrap();
            assert_eq!(report.request_counts.completed_requests, 2);
            // Input-only forecasting deliberately preserves native output KV:
            // a later input can still match blocks produced by decoding.
            assert_eq!(
                report.per_request[1].admission_history[0].reused_input_tokens, 8,
                "{backend:?}, {use_belady}",
            );
            assert_eq!(report.committed_prefill_tokens, 4);
        }
    }
}

#[test]
fn unsupported_forecast_inputs_fail_before_execution() {
    for backend in BACKENDS {
        let base = belady(spec(backend, 1, vec![request(0, vec![1; 16], 1)]));
        let mut closed_loop = base.clone();
        closed_loop.max_in_flight = Some(1);
        let mut length_only = base.clone();
        length_only.requests[0].input_token_ids = None;
        let mut no_cache = base.clone();
        no_cache.engine["rank"]["enable_prefix_caching"] = json!(false);
        let mut disaggregated = base;
        disaggregated.topology = ReplayTopology::Disaggregated {
            prefill: WorkerPoolSpec::default(),
            decode: WorkerPoolSpec::default(),
            handoff_latency_ms: 0.0,
        };
        for input in [closed_loop, length_only, no_cache, disaggregated] {
            let error = run(input).unwrap_err();
            assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
            assert!(
                error.to_string().to_lowercase().contains("belady"),
                "{error}"
            );
        }
    }
}
