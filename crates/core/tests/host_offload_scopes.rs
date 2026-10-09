// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-level G2 ownership: private per DP rank or one cluster-shared pool
//! across ranks, replicas and P/D roles.

use std::num::NonZeroU32;

use aisimulate_core::engine::generalized::{EngineIdentity, SchedulerCommand};
use aisimulate_core::engine::{
    Admission, Command, Engine, EngineConfig, EngineFactory, KvEventData, KvEventTier, Request,
    SharedG2Pool,
};
use aisimulate_core::replay::{
    G2DomainStats, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineFactory, ReplayReport,
    ReplaySpec, Replayer,
};
use serde_json::{Value, json};
use uuid::Uuid;

/// 9-token prompts: two complete 4-token blocks can be restored from G2.
const PROMPT: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

fn rank(host_offload: Value) -> Value {
    json!({
        "num_gpu_blocks": 16, "block_size": 4, "max_num_seqs": 4,
        "max_num_batched_tokens": 64, "kv_cache_bytes_per_token": 250_000,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        "native_host_offload": host_offload,
    })
}

fn host(scope: &str) -> Value {
    json!({"scope": scope, "num_host_blocks": 8, "kv_layout_id": "test-tp1"})
}

/// Identical prompts, each after the previous one's store landed. Round-robin
/// alternates replicas; `rank_field` then selects DP rank 0 twice, then 1.
fn requests(prompts: usize, rank_field: &str) -> Vec<Value> {
    (0..prompts)
        .map(|index| {
            let mut request = json!({
                "id": format!("r{index}"), "arrival_time_ms": 100.0 * index as f64,
                "input_tokens": PROMPT.len(), "input_token_ids": PROMPT, "output_tokens": 1,
            });
            request[rank_field] = json!(index / 2 % 2);
            request
        })
        .collect()
}

fn run(topology: Value, engine: Value, requests: Vec<Value>) -> anyhow::Result<ReplayReport> {
    run_with(&ReplayEngineFactory::new(), topology, engine, requests)
}

fn run_with(
    factory: &ReplayEngineFactory,
    topology: Value,
    engine: Value,
    requests: Vec<Value>,
) -> anyhow::Result<ReplayReport> {
    let spec: ReplaySpec = serde_json::from_value(json!({
        "version": 1, "topology": topology, "engine": engine,
        "requests": requests, "record_per_request": true,
    }))?;
    Ok(Replayer::new(spec, factory.clone())?
        .with_capture_options(ReplayCaptureOptions {
            determinism: ReplayDeterminism::CanonicalV1,
            ..ReplayCaptureOptions::default()
        })
        .run()?)
}

fn host_reuse(report: &ReplayReport) -> Vec<Option<usize>> {
    let mut records = report.per_request.iter().collect::<Vec<_>>();
    records.sort_by_key(|record| record.request_id.clone());
    records
        .iter()
        .map(|record| record.first_admission_host_reused_input_tokens)
        .collect()
}

fn aggregated(workers: usize) -> Value {
    json!({"kind": "aggregated", "workers": {"initial_workers": workers}})
}

fn disaggregated(prefill: usize) -> Value {
    json!({
        "kind": "disaggregated", "handoff_latency_ms": 0.0,
        "prefill": {"initial_workers": prefill}, "decode": {"initial_workers": 1},
    })
}

#[test]
fn attention_dp_ranks_share_g2_only_in_cluster_scope() {
    // Two replicas of ADP2: four initially cold G1 caches.
    for (scope, reuse, domains) in [
        ("dp_rank_local", [0, 0, 0, 0], vec![]),
        (
            "cluster_shared",
            [0, 8, 8, 8],
            vec![G2DomainStats {
                capacity_blocks: 8,
                resident_blocks: 2,
                used_blocks: 2,
            }],
        ),
    ] {
        let engine = json!({"dp_size": 2, "rank": rank(host(scope))});
        let report = run(aggregated(2), engine, requests(4, "dp_rank")).unwrap();
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(host_reuse(&report), reuse.map(Some), "{scope}");
        assert_eq!(report.g2_domains, domains, "{scope}");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json.get("g2_domains").is_some(), scope == "cluster_shared");
    }
}

#[test]
fn token_only_prefill_and_decode_join_one_cluster_pool() {
    for (prefill_scope, decode_scope, reuse) in [
        ("dp_rank_local", "dp_rank_local", [0, 0]),
        ("cluster_shared", "cluster_shared", [0, 8]),
        // Roles choose scopes independently; the shared prefill pool still reuses.
        ("cluster_shared", "dp_rank_local", [0, 8]),
    ] {
        let role = |scope| json!({"dp_size": 2, "rank": rank(host(scope))});
        let engine = json!({
            "rank": rank(Value::Null), "prefill": role(prefill_scope), "decode": role(decode_scope),
        });
        let report = run(disaggregated(2), engine, requests(2, "prefill_dp_rank")).unwrap();
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(
            host_reuse(&report),
            reuse.map(Some),
            "{prefill_scope}/{decode_scope}"
        );
    }
}

#[test]
fn each_replay_starts_with_a_cold_cluster_pool() {
    // One factory drives two replays of each deployment; the second must not
    // inherit G2 blocks the first stored.
    let factory = ReplayEngineFactory::new();
    let shared = || json!({"dp_size": 2, "rank": rank(host("cluster_shared"))});
    for (topology, engine, rank_field, reuse) in [
        (aggregated(2), shared(), "dp_rank", vec![0, 8, 8, 8]),
        (
            disaggregated(2),
            json!({"rank": rank(Value::Null), "prefill": shared(), "decode": shared()}),
            "prefill_dp_rank",
            vec![0, 8],
        ),
    ] {
        let trace = requests(reuse.len(), rank_field);
        let replay = || {
            let report = run_with(&factory, topology.clone(), engine.clone(), trace.clone())
                .unwrap()
                .with_wall_time_ms(0.0);
            (
                host_reuse(&report),
                serde_json::to_value(&report).unwrap(),
                serde_json::to_value(&report.per_request).unwrap(),
            )
        };
        let first = replay();
        assert_eq!(first.0, reuse.into_iter().map(Some).collect::<Vec<_>>());
        assert_eq!(replay(), first);
    }
}

#[test]
fn incompatible_shared_roles_and_unsupported_g3_topologies_fail_explicitly() {
    let g3 = json!({"scope": "worker_local", "num_g3_blocks": 8});
    let with_g3 = |mut rank: Value| {
        rank["g3_offload"] = g3.clone();
        rank
    };
    let mut other_layout = host("cluster_shared");
    other_layout["kv_layout_id"] = json!("test-tp2");
    for (topology, engine, expected) in [
        (
            disaggregated(1),
            json!({
                "rank": rank(Value::Null),
                "prefill": {"rank": rank(host("cluster_shared"))},
                "decode": {"rank": rank(other_layout)},
            }),
            "cluster_shared host_offload participants are incompatible",
        ),
        (
            aggregated(1),
            json!({"dp_size": 2, "rank": with_g3(rank(host("dp_rank_local")))}),
            "g3_offload supports only dp_size=1",
        ),
        (
            disaggregated(1),
            json!({"rank": with_g3(rank(host("dp_rank_local")))}),
            "g3_offload supports only aggregated replay",
        ),
    ] {
        let error = format!(
            "{:#}",
            run(topology, engine, requests(1, "dp_rank")).unwrap_err()
        );
        assert!(error.contains(expected), "{error}");
    }
}

/// A minimized-regression rank: 4-token blocks, 250 KB/token, fixed timing.
fn small_rank(gpu_blocks: u32, max_seqs: u32, max_batched: u32, extra: Value) -> Value {
    let mut rank = json!({
        "num_gpu_blocks": gpu_blocks, "max_num_seqs": max_seqs,
        "max_num_batched_tokens": max_batched, "block_size": 4,
        "kv_cache_bytes_per_token": 250_000,
        "timing_model": {"type": "fixed", "prefill_ms": 0.5, "decode_ms": 0.25},
    });
    rank.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    rank
}

/// Requests `(id, arrival_ms, first_token, input_tokens, output_tokens)`
/// whose prompt is the contiguous token range starting at `first_token`.
fn token_ranges(rows: &[(&str, f64, u32, u32, u32)]) -> Vec<Value> {
    rows.iter()
        .map(|&(id, arrival_ms, first, input_tokens, output_tokens)| {
            json!({
                "id": id, "arrival_time_ms": arrival_ms, "input_tokens": input_tokens,
                "input_token_ids": (first..first + input_tokens).collect::<Vec<_>>(),
                "output_tokens": output_tokens,
            })
        })
        .collect()
}

fn terminal_times(report: &ReplayReport) -> Vec<(Option<&str>, f64)> {
    let mut times = report
        .per_request
        .iter()
        .map(|record| (record.request_id.as_deref(), record.terminal_time_ms))
        .collect::<Vec<_>>();
    times.sort_by(|a, b| a.0.cmp(&b.0));
    times
}

#[test]
fn decode_g2_keeps_minimized_pd_inputs_live() {
    // A decode rank with private G2: previously a restore waiting for capacity
    // starved handoffs that already owned their KV (a replay dead end), and
    // handoff KV could land in G1 blocks still being copied to G2 (a panic).
    for (report, expected) in [
        (
            run(
                json!({
                    "decode": {"initial_workers": 1},
                    "handoff_latency_ms": 0.5,
                    "kind": "disaggregated",
                    "prefill": {"initial_workers": 1},
                }),
                json!({
                    "decode": {"dp_size": 1, "rank": small_rank(12, 4, 32, json!({
                        "native_host_offload": {
                            "d2h_bandwidth_gbps": 0.5,
                            "h2d_bandwidth_gbps": 1.5,
                            "latency_to_first_byte_ms": 0.5,
                            "num_host_blocks": 4,
                        },
                    }))},
                    "prefill": {"dp_size": 1, "rank": small_rank(16, 2, 24, json!({}))},
                    "rank": small_rank(16, 4, 24, json!({})),
                }),
                token_ranges(&[
                    ("r4", 10.0, 600, 8, 3),
                    ("r5", 10.0, 500, 31, 1),
                    ("r6", 10.0, 1000, 8, 1),
                ]),
            )
            .unwrap(),
            [("r4", 16.5), ("r5", 16.0), ("r6", 16.0)],
        ),
        (
            run(
                json!({
                    "decode": {"initial_workers": 1},
                    "handoff_latency_ms": 0.5,
                    "kind": "disaggregated",
                    "prefill": {"initial_workers": 2},
                }),
                json!({
                    "decode": {"dp_size": 1, "rank": small_rank(12, 4, 24, json!({
                        "native_host_offload": {
                            "d2h_bandwidth_gbps": 1.0,
                            "h2d_bandwidth_gbps": 4.0,
                            "latency_to_first_byte_ms": 0.0,
                            "num_host_blocks": 4,
                        },
                    }))},
                    "prefill": {"dp_size": 2, "rank": small_rank(16, 6, 32, json!({
                        "native_host_offload": {
                            "d2h_bandwidth_gbps": 4.0,
                            "h2d_bandwidth_gbps": 4.0,
                            "latency_to_first_byte_ms": 0.0,
                            "num_host_blocks": 4,
                        },
                    }))},
                    "rank": small_rank(12, 4, 24, json!({})),
                }),
                token_ranges(&[
                    ("r146", 167.5, 800, 16, 1),
                    ("r147", 168.5, 600, 9, 0),
                    ("r148", 168.5, 1000, 31, 3),
                ]),
            )
            .unwrap(),
            [("r146", 168.75), ("r147", 169.5), ("r148", 174.0)],
        ),
    ] {
        let expected = expected.map(|(id, at)| (Some(id), at));
        assert_eq!(terminal_times(&report), expected);
    }
}

#[test]
fn g3_restore_thrash_ends_by_computing_without_further_restores() {
    // A 31-token prefix restored from G3 cannot fit the 4-block G2, so every
    // promotion evicts one needed earlier. This minimized input never finished
    // before; r111 now stops restoring after 1024 restore rounds in which its
    // rank makes no progress, and computes everything G1 does not hold.
    let report = run(
        json!({"kind": "aggregated", "workers": {"initial_workers": 1}}),
        json!({
            "rank": small_rank(16, 2, 32, json!({
                "g3_offload": {
                    "latency_to_first_byte_ms": 0.25,
                    "num_g3_blocks": 64,
                    "read_bandwidth_gbps": 0.5,
                    "scope": "cluster_shared",
                    "shared_read_bandwidth_gbps": 2.0,
                    "shared_write_bandwidth_gbps": 0.75,
                    "write_bandwidth_gbps": 0.5,
                },
                "native_host_offload": {"d2h_bandwidth_gbps": 1.0, "h2d_bandwidth_gbps": 0.0, "num_host_blocks": 4},
            })),
        }),
        token_ranges(&[
            ("r93", 2872.55, 1100, 12, 3),
            ("r97", 3035.964, 600, 31, 1),
            ("r99", 3084.905, 300, 24, 6),
            ("r100", 3111.521, 1100, 31, 3),
            ("r108", 3285.675, 300, 31, 2),
            ("r109", 3325.659, 1000, 5, 6),
            ("r110", 3341.429, 200, 16, 2),
            ("r111", 3376.429, 1100, 31, 3),
        ]),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 8);
    let r111 = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("r111"))
        .unwrap();
    assert_eq!(r111.first_admit_ms, Some(9776.429));
    assert_eq!(
        (
            r111.first_admission_g1_reused_input_tokens,
            r111.first_admission_host_reused_input_tokens
        ),
        (Some(4), Some(0))
    );
    let g3 = report.g3_offload.unwrap();
    assert_eq!((g3.bypassed_restores, g3.read.completed_jobs), (1, 1024));
}

#[test]
fn shared_g2_g3_chunked_recompute_of_a_preempted_request_completes() {
    // Minimized from a long-prompt shared G2 + G3 stress input. r7 is
    // preempted after its seventh output token fills block 4, then readmitted
    // with a 16-token prefix hit and a 3-token budget: the chunk 16..19 leaves
    // the final generated block unhashed, which debug builds used to reject.
    // r9 later restores 8 tokens from the shared G2 pool.
    let report = run(
        json!({"kind": "aggregated", "workers": {"initial_workers": 1}}),
        json!({
            "rank": small_rank(8, 4, 3, json!({
                "g3_offload": {
                    "num_g3_blocks": 64,
                    "read_bandwidth_gbps": 0.5,
                    "scope": "cluster_shared",
                    "write_bandwidth_gbps": 1.0,
                },
                "native_host_offload": {
                    "d2h_bandwidth_gbps": 1.0,
                    "h2d_bandwidth_gbps": 1.0,
                    "kv_layout_id": "tiny",
                    "num_host_blocks": 5,
                    "scope": "cluster_shared",
                },
            })),
        }),
        token_ranges(&[
            ("r0", 1.0, 0, 8, 7),
            ("r3", 5.5, 200, 7, 4),
            ("r4", 5.5, 100, 10, 6),
            ("r6", 11.5, 0, 3, 11),
            ("r7", 11.5, 100, 13, 10),
            ("r9", 15.5, 0, 11, 8),
        ]),
    )
    .unwrap();
    let expected = [
        ("r0", 4.0),
        ("r3", 9.25),
        ("r4", 11.0),
        ("r6", 16.0),
        ("r7", 16.75),
        ("r9", 23.85),
    ]
    .map(|(id, at)| (Some(id), at));
    assert_eq!(terminal_times(&report), expected);
    let r7 = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("r7"))
        .unwrap();
    assert_eq!((r7.readmission_count, r7.reused_input_tokens), (1, 16));
    assert_eq!(host_reuse(&report), [0, 0, 0, 0, 0, 8].map(Some));
    let g3 = report.g3_offload.unwrap();
    assert_eq!((g3.read.completed_jobs, g3.bypassed_restores), (1, 0));
}

/// Drive `engine` until request `id` completes, returning the prompt tokens
/// its first admission restored from G2.
fn serve_prompt(engine: &mut Engine, id: u128, now_ms: &mut f64) -> usize {
    let request_id = Uuid::from_u128(id);
    // A peer that advanced the shared pool makes this engine's internal work
    // due now; the engine requires it to be processed before a command.
    if engine
        .next_internal_deadline_ms()
        .is_some_and(|deadline_ms| deadline_ms <= *now_ms)
    {
        engine.process_internal_work(*now_ms).unwrap();
    }
    engine
        .apply_command_effects(
            SchedulerCommand::new(
                0,
                Command::Submit(Request {
                    images: Vec::new(),
                    request_id,
                    tokens: PROMPT.to_vec(),
                    max_output_tokens: 1,
                    output_token_ids: None,
                }),
            ),
            *now_ms,
        )
        .unwrap();
    let mut host_reused = None;
    let mut record = |admissions: &[Admission]| {
        for admission in admissions {
            if admission.request_id == request_id && host_reused.is_none() {
                host_reused = Some(
                    admission
                        .cache_tier_attribution
                        .map_or(0, |tiers| tiers.host_reused_input_tokens),
                );
            }
        }
    };
    for _ in 0..1_000 {
        if let Some(deadline_ms) = engine.next_internal_deadline_ms()
            && deadline_ms <= *now_ms
        {
            let effects = engine.process_internal_work(*now_ms).unwrap();
            for rank in &effects.by_rank {
                record(&rank.effects.admissions);
            }
        }
        let Some(started) = engine.execute_pass(*now_ms).unwrap() else {
            *now_ms = engine
                .next_internal_deadline_ms()
                .expect("an unfinished request must have pending internal work")
                .max(*now_ms);
            continue;
        };
        for rank in &started.by_rank {
            record(&rank.effects.admissions);
        }
        *now_ms = started.end_ms;
        let completed = engine.complete_pass(started.pass_id, *now_ms).unwrap();
        if completed.effects.by_rank.iter().any(|rank| {
            rank.effects
                .outputs
                .iter()
                .any(|output| output.request_id == request_id && output.completed)
        }) {
            return host_reused.expect("a completed request was admitted");
        }
    }
    panic!("request {id} did not complete");
}

/// Let `engine` finish every transfer it started, such as write-through D2H.
fn settle(engine: &mut Engine, now_ms: &mut f64) {
    for _ in 0..1_000 {
        let Some(deadline_ms) = engine.next_internal_deadline_ms() else {
            return;
        };
        *now_ms = now_ms.max(deadline_ms);
        engine.process_internal_work(*now_ms).unwrap();
    }
    panic!("host transfers did not settle");
}

#[test]
fn engines_from_separate_factories_share_one_g2_pool() {
    let config: EngineConfig = serde_json::from_value(rank(host("cluster_shared"))).unwrap();
    let build = |pool: Option<&SharedG2Pool>| {
        let factory = EngineFactory::new(config.clone()).unwrap();
        let factory = match pool {
            Some(pool) => factory.with_shared_g2_pool(pool, 1),
            None => factory,
        };
        factory
            .build(EngineIdentity::new(0), NonZeroU32::MIN)
            .unwrap()
    };
    let pool = SharedG2Pool::new();
    // An invalid participant must not fix the pool contract for later joiners.
    let error = EngineFactory::new(config.clone())
        .unwrap()
        .with_shared_g2_pool(&pool, 0)
        .build(EngineIdentity::new(0), NonZeroU32::MIN)
        .err()
        .expect("zero tensor parallelism is invalid");
    assert!(
        format!("{error:#}").contains("tensor_parallel_size must be positive"),
        "{error:#}"
    );
    assert_eq!(pool.occupancy(), None);
    let (mut producer, mut consumer) = (build(Some(&pool)), build(Some(&pool)));
    let mut now_ms = 0.0;
    assert_eq!(serve_prompt(&mut producer, 1, &mut now_ms), 0);
    settle(&mut producer, &mut now_ms);
    assert_eq!(pool.occupancy(), Some((8, 2, 2)));
    // The consumer never computed the prompt; both full blocks come from G2.
    assert_eq!(serve_prompt(&mut consumer, 2, &mut now_ms), 8);

    // Without a pool, cluster-shared ranks have no deployment to join.
    let error = EngineFactory::new(config.clone())
        .unwrap()
        .build(EngineIdentity::new(0), NonZeroU32::MIN)
        .err()
        .expect("an unbound cluster-shared rank must fail");
    assert!(format!("{error:#}").contains("cluster_shared"), "{error:#}");
}

#[test]
fn an_engine_joining_a_warm_shared_pool_publishes_its_residency() {
    let mut config: EngineConfig = serde_json::from_value(rank(host("cluster_shared"))).unwrap();
    config.emit_kv_events = true;
    let pool = SharedG2Pool::new();
    let build = || {
        EngineFactory::new(config.clone())
            .unwrap()
            .with_shared_g2_pool(&pool, 1)
            .build(EngineIdentity::new(0), NonZeroU32::MIN)
            .unwrap()
    };
    let mut producer = build();
    let mut now_ms = 0.0;
    serve_prompt(&mut producer, 1, &mut now_ms);
    settle(&mut producer, &mut now_ms);
    assert_eq!(pool.occupancy(), Some((8, 2, 2)));

    // The late joiner's residency snapshot must be due now, not wait for an
    // unrelated request or transfer.
    let mut consumer = build();
    assert!(!consumer.is_drained());
    let deadline_ms = consumer
        .next_internal_deadline_ms()
        .expect("a queued residency snapshot is internal work");
    assert!(deadline_ms <= now_ms, "{deadline_ms} > {now_ms}");
    let effects = consumer.process_internal_work(now_ms).unwrap();
    let host_pinned_blocks = effects
        .by_rank
        .iter()
        .flat_map(|rank| &rank.effects.kv_events)
        .filter(|event| event.tier == KvEventTier::HostPinned)
        .map(|event| match &event.data {
            KvEventData::Stored(stored) => stored.blocks.len(),
            KvEventData::Removed { .. } => panic!("the snapshot only stores"),
        })
        .sum::<usize>();
    assert_eq!(host_pinned_blocks, 2);
    assert!(consumer.next_internal_deadline_ms().is_none());
    assert!(consumer.is_drained());
}
