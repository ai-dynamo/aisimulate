// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic Weka qualification through the real aggregated engine consumers.

use crate::engine::{Backend, EngineConfig, KvEventData, TimingModelConfig};
use crate::replay::loadgen::{ValidatedAgenticGraph, WorkloadDriver, load_weka_agentic_graph};
use crate::replay::{
    ReplayAdapters, ReplayArtifactKvEventVisibility, ReplayArtifacts, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRequestPool, ReplayRuntimeInput, ReplaySpec,
    ReplayTerminalStatus, ReplayTopology, Replayer,
};

struct PrefixCase {
    name: &'static str,
    input_length: usize,
    hashes: [&'static [u64]; 2],
    shared_input_tokens: usize,
}

fn graph(case: &PrefixCase) -> ValidatedAgenticGraph {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.json");
    // Authored locally for this test. Distinct source requests keep missing
    // units and partial tails private despite identical supplied hash IDs.
    let requests = case
        .hashes
        .iter()
        .enumerate()
        .map(|(index, hashes)| {
            serde_json::json!({
                "t": index as f64,
                "type": "s",
                "model": "synthetic-model",
                "in": case.input_length,
                "out": 1,
                "hash_ids": hashes,
            })
        })
        .collect::<Vec<_>>();
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "id": "prefix-qualification",
            "models": ["synthetic-model"],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": requests,
        }))
        .unwrap(),
    )
    .unwrap();
    load_weka_agentic_graph(&path, Some(64)).unwrap()
}

fn run(
    graph: &ValidatedAgenticGraph,
    backend: Backend,
    block_size: usize,
    caching: bool,
) -> (ReplayReport, ReplayArtifacts) {
    let engine = ReplayEngineConfig {
        rank: EngineConfig {
            backend,
            block_size,
            num_gpu_blocks: 256,
            enable_prefix_caching: caching,
            emit_kv_token_ids: true,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::for_backend(backend)
        },
        ..ReplayEngineConfig::default()
    };
    let spec = ReplaySpec {
        version: 1,
        topology: ReplayTopology::aggregated(1),
        engine: serde_json::to_value(engine).unwrap(),
        adapters: ReplayAdapters::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    let driver = WorkloadDriver::new_agentic_trace(graph.clone(), block_size).unwrap();
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap()
}

fn qualify(case: PrefixCase) {
    let graph = graph(&case);
    assert_eq!(graph.node_count(), 2);
    for backend in [Backend::Vllm, Backend::Sglang] {
        for block_size in [16, 32, 48, 64, 96, 128, 256] {
            for caching in [false, true] {
                let context = format!(
                    "{} backend={backend:?} block_size={block_size} caching={caching}",
                    case.name
                );
                let (report, artifacts) = run(&graph, backend, block_size, caching);
                assert_eq!(report.request_counts.completed_requests, 2, "{context}");
                assert_eq!(report.per_request.len(), 2, "{context}");
                assert_eq!(artifacts.requests.len(), 2, "{context}");

                let cold = &artifacts.requests[0];
                let warm = &artifacts.requests[1];
                let cold_hashes = cold.replay_hashes.as_ref().unwrap();
                let warm_hashes = warm.replay_hashes.as_ref().unwrap();
                let expected_shared_blocks = case.shared_input_tokens / block_size;
                assert_eq!(
                    cold_hashes.sequence_hashes.len(),
                    case.input_length / block_size,
                    "{context} only complete engine blocks have identities"
                );
                assert_eq!(
                    cold_hashes
                        .sequence_hashes
                        .iter()
                        .zip(&warm_hashes.sequence_hashes)
                        .take_while(|(cold, warm)| cold == warm)
                        .count(),
                    expected_shared_blocks,
                    "{context} source identities must retain exactly their shared prefix"
                );

                // Both schedulers recompute at least the final prompt token;
                // a fully shared prompt therefore has an admission cap below
                // its router-visible complete-block overlap.
                let expected_warm_reuse = if caching {
                    case.shared_input_tokens.min(case.input_length - 1) / block_size * block_size
                } else {
                    0
                };
                for (artifact, expected_reuse) in [(cold, 0), (warm, expected_warm_reuse)] {
                    let record = report
                        .per_request
                        .iter()
                        .find(|record| record.uuid == artifact.request_id.to_string())
                        .unwrap();
                    let node = graph
                        .nodes()
                        .iter()
                        .find(|node| Some(node.request_id()) == record.request_id.as_deref())
                        .unwrap();
                    assert_eq!(record.play_id.as_deref(), Some(node.play_id()), "{context}");
                    assert_eq!(record.input_length, case.input_length, "{context}");
                    assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
                    assert_eq!(record.admission_count, 1, "{context}");
                    assert_eq!(record.readmission_count, 0, "{context}");
                    assert_eq!(record.admission_history.len(), 1, "{context}");
                    let admission = &record.admission_history[0];
                    assert_eq!(admission.pool, ReplayRequestPool::Agg, "{context}");
                    assert!(!admission.is_readmission, "{context}");
                    assert_eq!(admission.reused_input_tokens, expected_reuse, "{context}");
                    assert_eq!(record.reused_input_tokens, expected_reuse, "{context}");
                    assert_eq!(record.first_admit_ms, Some(admission.at_ms), "{context}");
                    let output_cached_tokens = artifacts
                        .outputs
                        .iter()
                        .find(|output| output.request_id == artifact.request_id)
                        .unwrap()
                        .cached_tokens;
                    if backend == Backend::Vllm {
                        assert_eq!(output_cached_tokens, Some(expected_reuse), "{context}");
                    } else {
                        // SGLang reports reuse on Admission; its output event
                        // intentionally leaves this optional field absent.
                        assert_eq!(output_cached_tokens, None, "{context}");
                    }
                    assert_eq!(record.routing_history.len(), 1, "{context}");
                    assert_eq!(
                        record.routing_history[0].reported_overlap_tokens,
                        Some(0),
                        "{context} round-robin placement does not predict cache reuse"
                    );
                }
                let expected_ratio = expected_warm_reuse as f64 / (2 * case.input_length) as f64;
                assert_eq!(
                    report.first_admission_prefix_cache_reused_ratio, expected_ratio,
                    "{context} summary must use the observed first engine admission"
                );
                assert_eq!(
                    report.prefix_cache_reused_ratio, expected_ratio,
                    "{context}"
                );

                if caching {
                    // Read native cache stores before the warm request even
                    // exists. This reaches independent engine hashing instead
                    // of comparing two ReplayRequestHashes helper calls.
                    let stored = artifacts
                        .kv_events
                        .iter()
                        .filter(|event| event.observed_at_ms < warm.observed_at_ms)
                        .filter_map(|event| match &event.event.data {
                            KvEventData::Stored(stored) => Some(stored.blocks.iter()),
                            KvEventData::Removed { .. } => None,
                        })
                        .flatten()
                        .collect::<Vec<_>>();
                    for (index, sequence_hash) in cold_hashes.sequence_hashes.iter().enumerate() {
                        let native = stored
                            .iter()
                            .find(|block| block.block_hash == *sequence_hash)
                            .unwrap_or_else(|| {
                                panic!("{context} missing native store for cold block {index}")
                            });
                        assert_eq!(
                            native.tokens_hash, cold_hashes.local_block_hashes[index],
                            "{context} cold block {index} local hash"
                        );
                        assert_eq!(
                            native.token_ids.as_ref().map(Vec::len),
                            Some(block_size),
                            "{context} cold block {index} token extent"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn weka_full_prompt_reuse_obeys_first_admission_last_token_cap() {
    qualify(PrefixCase {
        name: "complete source units",
        input_length: 384,
        hashes: [&[10, 20, 30, 40, 50, 60]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn weka_partial_source_tail_stays_private_in_smaller_engine_blocks() {
    qualify(PrefixCase {
        name: "private partial tail",
        input_length: 416,
        hashes: [&[10, 20, 30, 40, 50, 60, 70]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn weka_missing_source_units_stay_private_at_actual_admission() {
    qualify(PrefixCase {
        name: "private missing full unit and partial tail",
        input_length: 449,
        hashes: [&[10, 20, 30, 40, 50, 60]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn weka_excess_source_hashes_do_not_create_extra_cache_reuse() {
    qualify(PrefixCase {
        name: "excess source hashes",
        input_length: 385,
        hashes: [&[10, 20, 30, 40, 50, 60, 70, 80]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn weka_equal_local_blocks_after_a_fork_do_not_resume_cache_reuse() {
    qualify(PrefixCase {
        name: "divergent prefix with matching later local units",
        input_length: 513,
        hashes: [
            &[10, 20, 30, 40, 50, 60, 70, 80],
            &[10, 20, 30, 99, 50, 60, 70, 80],
        ],
        shared_input_tokens: 192,
    });
}
