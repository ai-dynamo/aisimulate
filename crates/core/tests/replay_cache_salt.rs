// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Modified behavioral adaptation of SGLang v0.5.22; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use aisim_core::engine::{Backend, EngineConfig, KvEventData, TimingModelConfig};
use aisim_core::replay::loadgen::ReplayRequestHashes;
use aisim_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ReplayArtifactKvEventVisibility, ReplayCaptureOptions,
    ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayRequest, ReplaySpec,
    ReplayTopology, Replayer, WorkerPoolSpec,
};

#[test]
fn cache_salt_isolates_reuse_in_both_topologies_and_aggregated_events() {
    for disagg in [false, true] {
        let tokens = (0..16).collect::<Vec<_>>();
        let salts = [Some("alpha"), Some("alpha"), Some("beta"), None, Some("")];
        let engine = ReplayEngineConfig {
            rank: EngineConfig {
                block_size: 4,
                num_gpu_blocks: 256,
                max_num_batched_tokens: 64,
                emit_kv_events: true,
                emit_kv_token_ids: !disagg,
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 1.0,
                },
                ..EngineConfig::for_backend(Backend::Sglang)
            },
            ..Default::default()
        };
        let requests = salts
            .iter()
            .enumerate()
            .map(|(index, salt)| {
                serde_json::from_value::<ReplayRequest>(serde_json::json!({
                    "id": format!("request-{index}"),
                    "arrival_time_ms": index as f64 * 100.0,
                    "input_tokens": tokens.len(),
                    "input_token_ids": tokens,
                    "output_tokens": 2,
                    "cache_salt": salt,
                }))
                .unwrap()
            })
            .collect();
        let spec = ReplaySpec {
            encoder: None,
            version: CURRENT_REPLAY_SPEC_VERSION,
            topology: if disagg {
                ReplayTopology::Disaggregated {
                    prefill: WorkerPoolSpec::default(),
                    decode: WorkerPoolSpec::default(),
                    handoff_latency_ms: 0.0,
                }
            } else {
                ReplayTopology::aggregated(1)
            },
            engine: serde_json::to_value(engine).unwrap(),
            adapters: Default::default(),
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests,
        };
        let replayer = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_capture_options(ReplayCaptureOptions {
                capture_per_request: true,
                determinism: ReplayDeterminism::CanonicalV1,
                ..Default::default()
            });
        let (report, artifacts) = if disagg {
            (replayer.run().unwrap(), None)
        } else {
            let (report, artifacts) = replayer
                .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
                .unwrap();
            (report, Some(artifacts))
        };
        assert_eq!(report.request_counts.completed_requests, salts.len());
        for (index, expected_reuse) in [0, 12, 0, 0, 12].into_iter().enumerate() {
            let id = format!("request-{index}");
            let record = report
                .per_request
                .iter()
                .find(|record| record.request_id.as_deref() == Some(&id))
                .unwrap();
            assert_eq!(
                record.first_admission_reused_input_tokens, expected_reuse,
                "disagg={disagg} {id}"
            );
        }
        let Some(artifacts) = artifacts else {
            continue;
        };
        let stores = artifacts
            .kv_events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvEventData::Stored(stored) => Some(stored),
                _ => None,
            })
            .collect::<Vec<_>>();
        for salt in [None, Some("alpha"), Some("beta")] {
            let expected = ReplayRequestHashes::from_tokens_with_cache_salt(&tokens, 4, salt);
            let matching = stores
                .iter()
                .filter(|stored| stored.cache_salt.as_deref() == salt)
                .collect::<Vec<_>>();
            assert!(
                !matching.is_empty(),
                "missing stores: disagg={disagg}, salt={salt:?}"
            );
            for stored in matching {
                for block in &stored.blocks {
                    let token_ids = block.token_ids.as_ref().unwrap();
                    let index = tokens
                        .chunks_exact(4)
                        .position(|page| page == token_ids)
                        .unwrap();
                    assert_eq!(block.tokens_hash, expected.local_block_hashes[index]);
                    assert_eq!(block.block_hash, expected.sequence_hashes[index]);
                }
            }
        }
        assert!(
            stores
                .iter()
                .all(|stored| stored.cache_salt.as_deref() != Some(""))
        );
    }
}
