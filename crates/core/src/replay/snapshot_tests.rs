// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic snapshot qualification through real aggregated engine consumers.

use std::collections::VecDeque;

use uuid::Uuid;

use crate::engine::{Backend, EngineConfig, KvEventData, TimingModelConfig};
use crate::replay::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticPlayStatus, AgenticSnapshotOptions, AgenticSourceProvenance,
    PreparedAgenticSnapshots, ReplayRequestHashes, ValidatedAgenticGraph, WorkloadDriver,
    load_weka_agentic_graph,
};
use crate::replay::{
    AgenticRuntimeIdentity, DirectRequest, ReplayAdapters, ReplayArtifactKvEventVisibility,
    ReplayArtifacts, ReplayEngineConfig, ReplayEngineFactory, ReplayPromptTokenSource,
    ReplayReport, ReplayRequestContext, ReplayRequestPool, ReplayRuntimeInput, ReplaySpec,
    ReplayTerminalStatus, ReplayTopology, Replayer,
};

const ENGINE_BLOCK_SIZES: [usize; 7] = [16, 32, 48, 64, 96, 128, 256];

fn run(
    input: ReplayRuntimeInput,
    backend: Backend,
    block_size: usize,
    caching: bool,
    prefill_ms: f64,
) -> (ReplayReport, ReplayArtifacts) {
    let engine = ReplayEngineConfig {
        rank: EngineConfig {
            block_size,
            num_gpu_blocks: 256,
            enable_prefix_caching: caching,
            emit_kv_token_ids: true,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms,
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
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(input)
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap()
}

struct PrefixCase {
    name: &'static str,
    input_length: usize,
    hashes: [&'static [u64]; 2],
    shared_input_tokens: usize,
}

fn weka_graph(case: &PrefixCase) -> ValidatedAgenticGraph {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.json");
    // Authored locally: the first request supplies the qualification primer,
    // and the second preserves the same source prefix or forks from it.
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
            "id": "snapshot-prefix-qualification",
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

fn request(
    ordinal: u128,
    tokens: Vec<u32>,
    identity: AgenticRuntimeIdentity,
    output_length: usize,
) -> DirectRequest {
    DirectRequest {
        tokens,
        max_output_tokens: output_length,
        output_token_ids: Some(vec![u32::MAX; output_length]),
        uuid: Some(Uuid::from_u128(ordinal)),
        arrival_timestamp_ms: Some((ordinal - 1) as f64 * 1000.0),
        replay_context: Some(ReplayRequestContext {
            authored_id: identity.request_id.clone(),
            session_id: Some(identity.conversation_id.clone()),
            turn_index: None,
            metadata: serde_json::Value::Null,
            prompt_token_source: ReplayPromptTokenSource::Materialized,
            agentic: Some(identity),
        }),
        ..Default::default()
    }
}

fn assert_native_prompt_hashes(
    artifacts: &ReplayArtifacts,
    hashes: &ReplayRequestHashes,
    tokens: &[u32],
    block_size: usize,
    context: &str,
) {
    let stores = artifacts
        .kv_events
        .iter()
        .filter(|event| event.observed_at_ms < 1000.0)
        .filter_map(|event| match &event.event.data {
            KvEventData::Stored(stored) => Some(stored.blocks.iter()),
            KvEventData::Removed { .. } => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(hashes.sequence_hashes.len(), tokens.len() / block_size);
    for (index, sequence_hash) in hashes.sequence_hashes.iter().enumerate() {
        let native = stores
            .iter()
            .find(|block| block.block_hash == *sequence_hash)
            .unwrap_or_else(|| panic!("{context} missing native store for primer block {index}"));
        assert_eq!(
            native.tokens_hash, hashes.local_block_hashes[index],
            "{context}"
        );
        assert_eq!(
            native.token_ids.as_deref(),
            Some(&tokens[index * block_size..(index + 1) * block_size]),
            "{context} native block {index} preserves the prepared prompt tokens"
        );
    }
}

fn qualify_prefix(case: PrefixCase) {
    let graph = weka_graph(&case);
    let primer_id = graph
        .nodes()
        .iter()
        .find(|node| node.request_id().ends_with(":outer:0"))
        .unwrap()
        .request_id()
        .to_string();
    let profile_id = graph
        .nodes()
        .iter()
        .find(|node| node.request_id().ends_with(":outer:1"))
        .unwrap()
        .request_id()
        .to_string();
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let play = prepared.context().prepare_play(0, 0, Some(500.0)).unwrap();
    let next_play = prepared.context().prepare_play(0, 1, Some(500.0)).unwrap();
    let primer_tokens = play
        .materialize_prefix(&primer_id, case.input_length)
        .unwrap();
    let profile_tokens = play
        .materialize_prefix(&profile_id, case.input_length)
        .unwrap();
    let next_play_tokens = next_play
        .materialize_prefix(&profile_id, case.input_length)
        .unwrap();
    assert_eq!(
        primer_tokens
            .iter()
            .zip(&profile_tokens)
            .take_while(|(left, right)| left == right)
            .count(),
        case.shared_input_tokens,
        "{} source identities preserve the intended shared prefix",
        case.name
    );
    assert!(
        profile_tokens
            .iter()
            .zip(&next_play_tokens)
            .all(|(a, b)| a != b),
        "{} a new play must not reuse any previous-play logical token IDs",
        case.name
    );
    assert_ne!(play.evidence().cache_id, next_play.evidence().cache_id);

    for backend in [Backend::Vllm, Backend::Sglang] {
        for block_size in ENGINE_BLOCK_SIZES {
            let primer_hashes = ReplayRequestHashes::from_tokens(&primer_tokens, block_size as u32);
            let profile_hashes =
                ReplayRequestHashes::from_tokens(&profile_tokens, block_size as u32);
            let next_play_hashes =
                ReplayRequestHashes::from_tokens(&next_play_tokens, block_size as u32);
            assert_eq!(
                primer_hashes
                    .sequence_hashes
                    .iter()
                    .zip(&profile_hashes.sequence_hashes)
                    .take_while(|(a, b)| a == b)
                    .count(),
                case.shared_input_tokens / block_size
            );
            assert_ne!(
                profile_hashes.sequence_hashes[0],
                next_play_hashes.sequence_hashes[0]
            );
            for caching in [false, true] {
                let context = format!(
                    "{} backend={backend:?} block_size={block_size} caching={caching}",
                    case.name
                );
                // Qualification-only explicit warmup: production snapshot
                // execution does not submit primers until AIC-1812.
                let requests = VecDeque::from([
                    request(
                        1,
                        primer_tokens.clone(),
                        play.identity(&primer_id).unwrap(),
                        0,
                    ),
                    request(
                        2,
                        profile_tokens.clone(),
                        play.identity(&profile_id).unwrap(),
                        1,
                    ),
                    request(
                        3,
                        next_play_tokens.clone(),
                        next_play.identity(&profile_id).unwrap(),
                        1,
                    ),
                ]);
                let (report, artifacts) = run(
                    ReplayRuntimeInput::Requests(requests),
                    backend,
                    block_size,
                    caching,
                    1.0,
                );
                assert_eq!(report.request_counts.completed_requests, 3, "{context}");
                assert_eq!(report.per_request.len(), 3, "{context}");
                let expected_warm = if caching {
                    case.shared_input_tokens.min(case.input_length - 1) / block_size * block_size
                } else {
                    0
                };
                for (ordinal, expected_reuse) in [(1, 0), (2, expected_warm), (3, 0)] {
                    let record = report
                        .per_request
                        .iter()
                        .find(|record| record.uuid == Uuid::from_u128(ordinal).to_string())
                        .unwrap();
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
                    assert_eq!(record.routing_history[0].reported_overlap_tokens, Some(0));
                }
                assert_eq!(
                    report.first_admission_prefix_cache_reused_ratio,
                    expected_warm as f64 / (3 * case.input_length) as f64,
                    "{context} reported first-admission reuse comes from real engine admissions"
                );
                if caching {
                    assert_native_prompt_hashes(
                        &artifacts,
                        &primer_hashes,
                        &primer_tokens,
                        block_size,
                        &context,
                    );
                }
            }
        }
    }
}

#[test]
fn snapshot_same_play_primer_hits_and_next_play_misses_at_native_admission() {
    qualify_prefix(PrefixCase {
        name: "complete source units",
        input_length: 384,
        hashes: [&[10, 20, 30, 40, 50, 60]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn snapshot_partial_source_tail_stays_private_with_smaller_engine_blocks() {
    qualify_prefix(PrefixCase {
        name: "partial source tail",
        input_length: 416,
        hashes: [&[10, 20, 30, 40, 50, 60, 70]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn snapshot_missing_source_units_stay_private_at_native_admission() {
    qualify_prefix(PrefixCase {
        name: "missing source unit and partial tail",
        input_length: 449,
        hashes: [&[10, 20, 30, 40, 50, 60]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn snapshot_excess_source_hashes_do_not_create_cache_reuse() {
    qualify_prefix(PrefixCase {
        name: "excess source hashes",
        input_length: 385,
        hashes: [&[10, 20, 30, 40, 50, 60, 70, 80]; 2],
        shared_input_tokens: 384,
    });
}

#[test]
fn snapshot_equal_local_blocks_after_a_fork_do_not_resume_cache_reuse() {
    qualify_prefix(PrefixCase {
        name: "prefix divergence with equal later local blocks",
        input_length: 513,
        hashes: [
            &[10, 20, 30, 40, 50, 60, 70, 80],
            &[10, 20, 30, 99, 50, 60, 70, 80],
        ],
        shared_input_tokens: 192,
    });
}

fn frontier_graph() -> ValidatedAgenticGraph {
    let row =
        |id: &str, session: &str, at_ms: f64, duration_ms: f64, hashes: &[u64], dependencies| {
            AgenticMooncakeRow {
                request_id: id.into(),
                play_id: "play".into(),
                session_id: session.into(),
                model: "synthetic-model".into(),
                input_length: Some(hashes.len() * 64),
                output_length: Some(1),
                hash_ids: Some(hashes.to_vec()),
                not_before_ms: at_ms,
                recorded_api_time_ms: Some(duration_ms),
                dependencies,
                ..Default::default()
            }
        };
    let dependency = |id: &str, trigger, relation, delay_ms| AgenticDependency {
        request_id: id.into(),
        trigger,
        relation,
        delay_ms,
    };
    ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 64,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored-test".into(),
                digest: "aic-1811-frontier-fixture".into(),
            },
        },
        vec![
            row("root", "root-session", 0.0, 799.0, &[10, 20], vec![]),
            row(
                "child",
                "child-session",
                800.0,
                100.0,
                &[10, 20, 30],
                vec![dependency(
                    "root",
                    AgenticDependencyTrigger::Dispatch,
                    AgenticDependencyRelation::Spawn,
                    800.0,
                )],
            ),
            row(
                "join",
                "root-session",
                1000.0,
                10.0,
                &[10, 20, 30, 40],
                vec![
                    dependency(
                        "root",
                        AgenticDependencyTrigger::Completion,
                        AgenticDependencyRelation::Sequence,
                        201.0,
                    ),
                    dependency(
                        "child",
                        AgenticDependencyTrigger::Completion,
                        AgenticDependencyRelation::Join,
                        100.0,
                    ),
                ],
            ),
        ],
    )
    .unwrap()
}

#[test]
fn snapshot_consumes_recorded_active_history_and_drains_live_child_join_without_primers() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        let prepared = frontier_graph()
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
            .unwrap();
        let play = prepared.context().prepare_play(0, 0, Some(500.0)).unwrap();
        let evidence = play.evidence();
        let historical = evidence
            .requests
            .iter()
            .find(|request| request.source_request_id == "root")
            .unwrap();
        assert!(historical.historical);
        assert!(historical.recorded_end_ms.unwrap() > evidence.t_star_ms);
        assert_eq!(evidence.primers.len(), 1);
        assert_eq!(evidence.primers[0].source_request_id, "root");
        assert_eq!(evidence.primers[0].input_length, 128);
        let child_id = play.identity("child").unwrap().request_id;
        let join_id = play.identity("join").unwrap().request_id;
        let historical_id = play.identity("root").unwrap().request_id;
        let child_hashes =
            ReplayRequestHashes::from_tokens(&play.materialize_prefix("child", 192).unwrap(), 64);
        let join_hashes =
            ReplayRequestHashes::from_tokens(&play.materialize_prefix("join", 256).unwrap(), 64);
        let driver = WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![play]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap();
        // Actual child service is deliberately slower than its recorded 100 ms.
        // The live join must wait for real completion plus its remaining delay.
        let (report, artifacts) = run(
            ReplayRuntimeInput::Workload(driver),
            backend,
            64,
            true,
            500.0,
        );
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_input_tokens, 192 + 256);
        assert_eq!(artifacts.requests.len(), 2);
        assert_eq!(
            artifacts.requests[0].replay_hashes.as_ref(),
            Some(&child_hashes)
        );
        assert_eq!(
            artifacts.requests[1].replay_hashes.as_ref(),
            Some(&join_hashes)
        );
        assert!(
            report
                .per_request
                .iter()
                .all(|request| request.request_id.as_deref() != Some("root"))
        );
        let child = report
            .per_request
            .iter()
            .find(|request| request.request_id.as_deref() == Some("child"))
            .unwrap();
        let join = report
            .per_request
            .iter()
            .find(|request| request.request_id.as_deref() == Some("join"))
            .unwrap();
        assert_eq!(child.agentic.as_ref().unwrap().request_id, child_id);
        assert_eq!(join.agentic.as_ref().unwrap().request_id, join_id);
        assert_eq!(child.arrival_time_ms, 300.0);
        assert!(child.terminal_time_ms >= child.arrival_time_ms + 500.0);
        assert_eq!(join.arrival_time_ms, child.terminal_time_ms + 100.0);
        assert!(join.terminal_time_ms >= join.arrival_time_ms + 500.0);
        assert_eq!(
            child.admission_history[0].reused_input_tokens, 0,
            "snapshot descriptors do not physically prime the engine"
        );
        assert_eq!(join.admission_history[0].reused_input_tokens, 192);
        assert_eq!(
            report.first_admission_prefix_cache_reused_ratio,
            192.0 / 448.0
        );
        let outcomes = report.agentic_play_outcomes.as_ref().unwrap();
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, AgenticPlayStatus::Completed);
        let lifecycle = report.agentic_lifecycle.as_ref().unwrap();
        assert!(
            lifecycle
                .events
                .iter()
                .all(|event| event.request_id.as_deref() != Some("root"))
        );
        assert!(
            report
                .per_request
                .iter()
                .all(|request| { request.agentic.as_ref().unwrap().request_id != historical_id })
        );
    }
}

#[test]
fn snapshot_seeded_initial_lanes_use_separate_cache_identities_in_the_same_engine() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        let prepared = frontier_graph()
            .prepare_snapshots(2, AgenticSnapshotOptions { seed: 2026 })
            .unwrap();
        let evidence = prepared.snapshots().to_vec();
        assert_eq!(evidence.len(), 2);
        assert_ne!(evidence[0].cache_id, evidence[1].cache_id);
        assert_ne!(evidence[0].play_id, evidence[1].play_id);
        let driver = WorkloadDriver::new_agentic_snapshots(prepared, 64, true, 1.0).unwrap();
        let (report, artifacts) = run(ReplayRuntimeInput::Workload(driver), backend, 64, true, 1.0);
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(artifacts.requests.len(), 4);
        for snapshot in &evidence {
            assert!((250.0..750.0).contains(&snapshot.t_star_ms));
            for (source_id, expected_reuse) in [("child", 0), ("join", 192)] {
                let request = snapshot
                    .requests
                    .iter()
                    .find(|request| request.source_request_id == source_id)
                    .unwrap();
                let record = report
                    .per_request
                    .iter()
                    .find(|record| record.agentic.as_ref() == Some(&request.identity))
                    .unwrap();
                assert_eq!(record.request_id.as_deref(), Some(source_id));
                assert_eq!(record.play_id.as_deref(), Some(snapshot.play_id.as_str()));
                assert_eq!(
                    record.arrival_time_ms,
                    request.recorded_start_ms - snapshot.t_star_ms
                );
                assert_eq!(record.admission_history.len(), 1);
                assert_eq!(
                    record.admission_history[0].reused_input_tokens,
                    expected_reuse
                );
            }
        }
        assert_eq!(
            report.first_admission_prefix_cache_reused_ratio,
            192.0 / 448.0
        );
        assert!(
            report
                .agentic_play_outcomes
                .as_ref()
                .unwrap()
                .iter()
                .all(|play| play.status == AgenticPlayStatus::Completed)
        );
    }
}

#[test]
fn snapshot_request_start_at_the_cut_stays_live_and_matches_turn_zero_tokens() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        let prepared = frontier_graph()
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
            .unwrap();
        let zero = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
        let cut = prepared.context().prepare_play(0, 0, Some(800.0)).unwrap();
        for (source_id, length) in [("root", 128), ("child", 192), ("join", 256)] {
            assert_eq!(
                zero.materialize_prefix(source_id, length).unwrap(),
                cut.materialize_prefix(source_id, length).unwrap()
            );
            assert_eq!(
                zero.identity(source_id).unwrap(),
                cut.identity(source_id).unwrap()
            );
        }
        let child_id = cut.identity("child").unwrap().request_id;
        let child = cut
            .evidence()
            .requests
            .iter()
            .find(|request| request.source_request_id == "child")
            .unwrap();
        assert!(!child.historical);
        assert_eq!(child.remaining_delay_ms, 0.0);
        let driver = WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![cut]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap();
        let (report, _) = run(
            ReplayRuntimeInput::Workload(driver),
            backend,
            64,
            true,
            500.0,
        );
        assert_eq!(report.request_counts.completed_requests, 2);
        let child = report
            .per_request
            .iter()
            .find(|request| {
                request
                    .agentic
                    .as_ref()
                    .map(|identity| identity.request_id.as_str())
                    == Some(child_id.as_str())
            })
            .unwrap();
        assert_eq!(child.arrival_time_ms, 0.0);
        assert_eq!(child.first_admit_ms, Some(0.0));
        assert!(child.terminal_time_ms >= 500.0);
    }
}

#[test]
fn snapshot_original_prompt_has_fixed_token_and_native_hash_vectors() {
    let graph = weka_graph(&PrefixCase {
        name: "literal native hash qualification",
        input_length: 257,
        hashes: [&[u64::MAX, 0, 1_u64 << 32, 7, u64::MAX - 1]; 2],
        shared_input_tokens: 256,
    });
    let source_id = graph
        .nodes()
        .iter()
        .find(|node| node.request_id().ends_with(":outer:0"))
        .unwrap()
        .request_id()
        .to_string();
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let play = prepared.context().prepare_play(0, 0, Some(500.0)).unwrap();
    // This one original prompt introduces four complete source units and one
    // private tail. These are encounter-assigned IDs in the retained context;
    // the fixture makes no stability claim across different corpus orderings.
    let mut expected_tokens = [vec![0; 64], vec![1; 64], vec![2; 64], vec![3; 64]].concat();
    expected_tokens.push(4);
    let tokens = play.materialize_prefix(&source_id, 257).unwrap();
    assert_eq!(tokens, expected_tokens);

    // Independently checked against the system XXH3 library: little-endian
    // u32 tokens, seed 1337; each subsequent sequence hashes (parent, local).
    let goldens = [
        (
            16,
            [
                vec![5523182284766269584; 4],
                vec![13518392937921988335; 4],
                vec![4954898952464910565; 4],
                vec![18442881532971542694; 4],
            ]
            .concat(),
            vec![
                5523182284766269584,
                3102875163438626881,
                12015872284175039520,
                13491560310605905736,
                2753715453177872845,
                5853119159508116031,
                16358848901121747536,
                6171693240882175658,
                7711537130486916250,
                17852816118423437953,
                5708400198388538185,
                3174120619586616559,
                2390017692163517648,
                11138642071616995084,
                11588736381645634493,
                12314123547308073350,
            ],
        ),
        (
            32,
            vec![
                8078176463474809903,
                8078176463474809903,
                11345337670721771672,
                11345337670721771672,
                12481126450064483879,
                12481126450064483879,
                2671291174597333161,
                2671291174597333161,
            ],
            vec![
                8078176463474809903,
                18396072900361164560,
                1438722767750827987,
                13491503657893108256,
                7327062645801226586,
                12888739565981173968,
                7223061985546373822,
                13489329121857577087,
            ],
        ),
        (
            48,
            vec![
                5668416277218608172,
                14058210135089406444,
                6754173385707188877,
                9744416516818080641,
                10861956436164017793,
            ],
            vec![
                5668416277218608172,
                3476548722019456492,
                15586773222372520177,
                12324737480133932217,
                17869052026612087845,
            ],
        ),
        (
            64,
            vec![
                15480293642169978529,
                16951273711404654616,
                2439962053643786207,
                15232596819325015918,
            ],
            vec![
                15480293642169978529,
                16972514322484578542,
                7711429637831522643,
                15046190559102504818,
            ],
        ),
        (
            96,
            vec![5305219445779718178, 3717269782543760841],
            vec![5305219445779718178, 4743428074381706595],
        ),
        (
            128,
            vec![5853190029040270819, 8121016376841148077],
            vec![5853190029040270819, 13305251262611722040],
        ),
        (256, vec![3337819277574342568], vec![3337819277574342568]),
    ];
    for (block_size, local, sequence) in goldens {
        let expected = ReplayRequestHashes {
            local_block_hashes: local,
            sequence_hashes: sequence,
        };
        assert_eq!(
            ReplayRequestHashes::from_tokens(&tokens, block_size as u32),
            expected
        );
        for backend in [Backend::Vllm, Backend::Sglang] {
            let input = ReplayRuntimeInput::Requests(VecDeque::from([request(
                1,
                tokens.clone(),
                play.identity(&source_id).unwrap(),
                0,
            )]));
            let (report, artifacts) = run(input, backend, block_size, true, 1.0);
            assert_eq!(report.request_counts.completed_requests, 1);
            assert_native_prompt_hashes(
                &artifacts,
                &expected,
                &tokens,
                block_size,
                &format!("literal native hashes backend={backend:?} block_size={block_size}"),
            );
        }
    }
}

#[test]
fn snapshot_new_incarnation_can_finish_before_old_incarnation_without_identity_or_cache_leak() {
    let graph = weka_graph(&PrefixCase {
        name: "overlapping old and new incarnation",
        input_length: 257,
        hashes: [&[10, 20, 30, 40, 50]; 2],
        shared_input_tokens: 256,
    });
    let source_id = graph
        .nodes()
        .iter()
        .find(|node| node.request_id().ends_with(":outer:0"))
        .unwrap()
        .request_id()
        .to_string();
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let old = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
    let new = prepared.context().prepare_play(0, 1, Some(0.0)).unwrap();
    let old_identity = old.identity(&source_id).unwrap();
    let new_identity = new.identity(&source_id).unwrap();
    assert_eq!(old_identity.lane_id, new_identity.lane_id);
    assert_ne!(old_identity.play_id, new_identity.play_id);
    assert_ne!(old_identity.cache_id, new_identity.cache_id);
    assert_ne!(old_identity.request_id, new_identity.request_id);
    for backend in [Backend::Vllm, Backend::Sglang] {
        let mut old_request = request(
            1,
            old.materialize_prefix(&source_id, 257).unwrap(),
            old_identity.clone(),
            32,
        );
        let mut new_request = request(
            2,
            new.materialize_prefix(&source_id, 257).unwrap(),
            new_identity.clone(),
            1,
        );
        old_request.replay_context.as_mut().unwrap().authored_id = source_id.clone();
        new_request.replay_context.as_mut().unwrap().authored_id = source_id.clone();
        new_request.arrival_timestamp_ms = Some(2.0);
        let (report, _) = run(
            ReplayRuntimeInput::Requests(VecDeque::from([old_request, new_request])),
            backend,
            64,
            true,
            1.0,
        );
        assert_eq!(report.request_counts.completed_requests, 2);
        let old_record = report
            .per_request
            .iter()
            .find(|record| record.uuid == Uuid::from_u128(1).to_string())
            .unwrap();
        let new_record = report
            .per_request
            .iter()
            .find(|record| record.uuid == Uuid::from_u128(2).to_string())
            .unwrap();
        assert!(
            new_record.first_admit_ms.unwrap() < old_record.terminal_time_ms,
            "{backend:?} requests must overlap in the native engine"
        );
        assert!(
            new_record.terminal_time_ms < old_record.terminal_time_ms,
            "{backend:?} old completion must arrive after new completion"
        );
        for (record, identity, output_length) in [
            (old_record, &old_identity, 32),
            (new_record, &new_identity, 1),
        ] {
            assert_eq!(record.request_id.as_ref(), Some(&source_id));
            assert_eq!(record.agentic.as_ref(), Some(identity));
            assert_eq!(record.session_id.as_ref(), Some(&identity.conversation_id));
            assert_eq!(record.output_length, output_length);
            assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
            assert_eq!(record.admission_history.len(), 1);
            assert_eq!(record.admission_history[0].reused_input_tokens, 0);
        }
        assert_eq!(report.first_admission_prefix_cache_reused_ratio, 0.0);
    }
}
