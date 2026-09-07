// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::*;
use crate::engine::{EngineConfig, G3Scope, TimingModelConfig};
use crate::replay::components::NoReplayMetadata;
use crate::replay::core::NoEngineEvents;
use crate::replay::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::{
    ReplayArtifactKvEventVisibility, ReplaySpec, ReplayTelemetryObserver, ReplayTelemetrySnapshot,
    Replayer, WorkerStage, run_engine_replay,
};
use serde_json::{Value, json};

type RoundRobinAggRuntime =
    AggRuntimeImpl<AggregatedRoundRobinPlacement<()>, NoEngineEvents, NoReplayMetadata>;

struct NoopTelemetryObserver;

impl ReplayTelemetryObserver for NoopTelemetryObserver {
    fn on_sample(&mut self, _snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        Ok(())
    }
}

fn request(uuid: u128, arrival_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens: vec![1; 4],
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(uuid)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn runtime(pending: VecDeque<DirectRequest>) -> RoundRobinAggRuntime {
    let config = ReplayEngineConfig {
        rank: EngineConfig {
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            ..EngineConfig::default()
        },
        ..ReplayEngineConfig::default()
    };
    let role_factory = ReplayEngineFactory::new()
        .role_factory(&config, WorkerStage::Aggregated, false)
        .unwrap();
    RoundRobinAggRuntime::new_composed(
        role_factory,
        AdmissionQueue::new_requests(pending, ReplayMode::Trace),
        1,
        None,
        |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
    )
    .unwrap()
}

#[test]
fn telemetry_only_timestamps_do_not_enter_the_agg_semantic_drain() {
    let pending = VecDeque::from([request(1, 10.0)]);
    let (_, baseline_stats) = runtime(pending.clone()).run().unwrap();
    let (_, observed_stats) = runtime(pending)
        .with_telemetry_observer(1.0, Box::new(NoopTelemetryObserver))
        .run()
        .unwrap();

    assert!(baseline_stats.semantic_drain_count > 1);
    assert_eq!(
        observed_stats.semantic_drain_count, baseline_stats.semantic_drain_count,
        "telemetry-only heartbeats must not wake aggregate semantic replay work"
    );
}

struct MismatchedPlacement;

impl PlacementPolicy<ReplayRequestPayload> for MismatchedPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        _request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<crate::replay::PlacementEffects> {
        Ok(crate::replay::PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id: Uuid::from_u128(999),
                scheduler_id: 0,
                reported_overlap_tokens: 0,
                cache_sample: None,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _: (), _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

#[test]
fn mismatched_placement_does_not_retain_arrival_or_offered_traffic() {
    let role_factory = ReplayEngineFactory::new()
        .role_factory(
            &ReplayEngineConfig::default(),
            WorkerStage::Aggregated,
            false,
        )
        .unwrap();
    let mut runtime =
        AggRuntimeImpl::<MismatchedPlacement, NoEngineEvents, NoReplayMetadata>::new_composed(
            role_factory,
            AdmissionQueue::new_requests(VecDeque::new(), ReplayMode::Trace),
            1,
            None,
            |_, _| Ok(MismatchedPlacement),
        )
        .unwrap()
        .into_steppable();
    let uuid = Uuid::from_u128(1);
    let error = runtime.submit_dynamic(request(1, 0.0)).unwrap_err();
    assert!(error.to_string().contains("while placing"), "{error}");
    assert!(!runtime.collector.contains_request(uuid));
    assert_eq!(runtime.traffic.drain_planner(1_000.0).num_req, 0);
    assert!(runtime.requests.is_empty());
    assert_eq!(runtime.cluster_in_flight(), 0);
    assert!(runtime.cancel_dynamic(uuid).unwrap().is_none());
    assert_eq!(
        runtime
            .take_report_dynamic(0.0)
            .unwrap()
            .request_counts
            .num_requests,
        0
    );
}

// G3 fixtures independently adapt vLLM behavioral contracts (Apache-2.0).
// Upstream: https://github.com/vllm-project/vllm/tree/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
// Paths: vllm/v1/core/sched/{scheduler,request_queue}.py and
// vllm/v1/kv_offload/tiering/manager.py. See THIRD_PARTY_NOTICES.md.
fn g3_spec(
    scope: G3Scope,
    workers: usize,
    g1: usize,
    g2: usize,
    block: usize,
    rows: &[(f64, u32, usize, usize)],
) -> ReplaySpec {
    let requests = rows.iter().enumerate().map(|(i, &(arrival, base, count, output))| json!({
        "id": i.to_string(), "arrival_time_ms": arrival, "input_tokens": count,
        "input_token_ids": (base..base+count as u32).collect::<Vec<_>>(), "output_tokens": output,
    })).collect::<Vec<_>>();
    serde_json::from_value(json!({"topology":{"kind":"aggregated","workers":{"initial_workers":workers}},
        "engine":{"rank":{"num_gpu_blocks":g1,"block_size":block,"max_num_seqs":3,
            "max_num_batched_tokens":128,"enable_prefix_caching":true,"kv_cache_bytes_per_token":256,
            "native_host_offload":{"num_host_blocks":g2,"d2h_bandwidth_gbps":0.0,"h2d_bandwidth_gbps":0.0},
            "g3_offload":{"scope":scope,"num_g3_blocks":8,"latency_to_first_byte_ms":0.0,
                "read_bandwidth_gbps":0.0,"write_bandwidth_gbps":0.0,
                "shared_read_bandwidth_gbps":0.0,"shared_write_bandwidth_gbps":0.0},"timing_model":{"type":"fixed","prefill_ms":0.0,"decode_ms":0.0}}},
        "requests":requests})).unwrap()
}

fn g3_patch(spec: ReplaySpec, update: impl FnOnce(&mut Value)) -> ReplaySpec {
    let mut value = serde_json::to_value(spec).unwrap();
    update(&mut value);
    serde_json::from_value(value).unwrap()
}

fn g3_runtime(input: ReplaySpec) -> RoundRobinAggRuntime {
    let config: ReplayEngineConfig = serde_json::from_value(input.engine).unwrap();
    let factory = ReplayEngineFactory::new()
        .role_factory(&config, WorkerStage::Aggregated, false)
        .unwrap();
    let pending = input
        .requests
        .into_iter()
        .enumerate()
        .map(|(index, row)| DirectRequest {
            tokens: row.input_token_ids.unwrap(),
            max_output_tokens: row.output_tokens,
            uuid: Some(Uuid::from_u128(index as u128)),
            arrival_timestamp_ms: Some(row.arrival_time_ms),
            ..DirectRequest::default()
        })
        .collect();
    RoundRobinAggRuntime::new_composed(
        factory,
        AdmissionQueue::new_requests(pending, ReplayMode::Trace),
        1,
        None,
        |dp_size, topology| Ok(AggregatedRoundRobinPlacement::new(dp_size, topology)),
    )
    .unwrap()
}

#[test]
fn g3_scaling_drains_writes_and_reads_then_reuses_only_shared_storage() {
    use crate::replay::scaling::{
        ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot,
    };
    use std::{cell::RefCell, rc::Rc};

    struct ScaleCycle(Rc<RefCell<Vec<Vec<usize>>>>);
    impl ReplayScalingPolicy for ScaleCycle {
        fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
            Ok(1.0)
        }
        fn on_tick(
            &mut self,
            snapshot: ReplayScalingSnapshot,
        ) -> anyhow::Result<ReplayScalingDecision> {
            self.0.borrow_mut().push(snapshot.active_decode_ids.clone());
            let (target, next) = match snapshot.tick_ordinal {
                0 => (0, Some(20.0)),
                1 => {
                    assert!(snapshot.draining_decode_ids.is_empty());
                    (1, Some(31.0))
                }
                2 => (0, Some(50.0)),
                3 => {
                    assert!(snapshot.draining_decode_ids.is_empty());
                    (1, None)
                }
                _ => unreachable!(),
            };
            Ok(ReplayScalingDecision {
                target_decode: Some(target),
                next_tick_ms: next,
                ..Default::default()
            })
        }
    }
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut runs = Vec::new();
        for _ in 0..2 {
            let input = g3_patch(
                g3_spec(
                    scope,
                    1,
                    4,
                    4,
                    4,
                    &[(0.0, 0, 9, 1), (30.0, 0, 9, 1), (60.0, 0, 9, 1)],
                ),
                |v| {
                    v["engine"]["rank"]["g3_offload"]["latency_to_first_byte_ms"] = json!(10.0);
                },
            );
            let seen = Rc::new(RefCell::new(Vec::new()));
            let (collector, _) = g3_runtime(input)
                .with_scaling_policy(Box::new(ScaleCycle(seen.clone())))
                .run()
                .unwrap();
            assert_eq!(*seen.borrow(), vec![vec![0], vec![], vec![1], vec![]]);
            let report = collector.finish();
            assert_eq!(report.request_counts.completed_requests, 3);
            let stats = report.g3_offload.unwrap();
            assert_eq!(stats.pending_blocks, 0);
            assert_eq!(stats.read.submitted_jobs, stats.read.completed_jobs);
            assert_eq!(stats.write.submitted_jobs, stats.write.completed_jobs);
            if scope == G3Scope::WorkerLocal {
                assert_eq!(stats.cross_worker_read_blocks, 0);
                assert_eq!(stats.read.completed_jobs, 0);
            } else {
                assert_eq!(stats.read.completed_jobs, 2);
                assert_eq!(stats.cross_worker_read_blocks, 4);
            }
            runs.push((stats, report.throughput.duration_ms));
        }
        assert_eq!(runs[0], runs[1]);
    }
}

#[test]
fn g3_fixed_scaling_policy_preserves_no_policy_results() {
    use crate::replay::scaling::NoScaling;
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let input = g3_spec(scope, 1, 4, 4, 4, &[(0.0, 0, 9, 1)]);
        let (plain, _) = g3_runtime(input.clone()).run().unwrap();
        let (fixed, _) = g3_runtime(input)
            .with_scaling_policy(Box::new(NoScaling))
            .run()
            .unwrap();
        let summaries = [plain, fixed].map(|collector| {
            let mut report = serde_json::to_value(collector.finish()).unwrap();
            for field in [
                "wall_time_ms",
                "processed_tokens_per_s",
                "processed_output_tokens_per_s",
            ] {
                report.as_object_mut().unwrap().remove(field);
            }
            report
        });
        assert_eq!(summaries[0], summaries[1]);
    }
}

#[test]
fn g3_reused_runtime_reports_cumulative_counters_and_keeps_cache() {
    fn submit_and_drain(runtime: &mut RoundRobinAggRuntime, id: u128, base: u32, count: u32) {
        runtime
            .submit_dynamic(DirectRequest {
                tokens: (base..base + count).collect(),
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(id)),
                arrival_timestamp_ms: Some(runtime.now_ms()),
                ..DirectRequest::default()
            })
            .unwrap();
        for _ in 0..1000 {
            runtime.step_dynamic_until(f64::INFINITY).unwrap();
            runtime.take_step_tokens();
            runtime.take_step_terminals();
            if runtime.is_workload_done() {
                return;
            }
        }
        panic!("request and cache I/O did not drain");
    }
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let mut runtime = g3_runtime(g3_spec(scope, 1, 4, 3, 4, &[])).into_steppable();
        submit_and_drain(&mut runtime, 1, 0, 9);
        submit_and_drain(&mut runtime, 2, 100, 15);
        let first = runtime.take_report_dynamic(0.0).unwrap();
        assert_eq!(first.request_counts.completed_requests, 2);
        let first = first.g3_offload.unwrap();
        assert!(first.write.completed_bytes > 0);
        assert_eq!(first.read.completed_jobs, 0);
        assert!(first.resident_blocks > 0);
        submit_and_drain(&mut runtime, 3, 0, 9);
        let second = runtime.take_report_dynamic(0.0).unwrap();
        assert_eq!(second.request_counts.completed_requests, 1);
        let second = second.g3_offload.unwrap();
        assert!(
            second.read.completed_bytes > 0,
            "second round restores the first round's G3 prefix"
        );
        assert_eq!(
            second.write, first.write,
            "reporting must not clear cache or counters"
        );
        assert_eq!(second.resident_blocks, first.resident_blocks);
        let idle = runtime.take_report_dynamic(0.0).unwrap();
        assert_eq!(idle.request_counts.num_requests, 0);
        assert_eq!(idle.g3_offload.unwrap(), second);
    }
}

#[test]
fn g3_h2d_artifacts_follow_logical_prefix_for_g2_and_g3() {
    for g3 in [false, true] {
        let input = g3_spec(
            G3Scope::WorkerLocal,
            1,
            4,
            if g3 { 3 } else { 8 },
            4,
            &[(0.0, 0, 9, 1), (10.0, 100, 15, 1), (20.0, 0, 9, 1)],
        );
        let input = g3_patch(input, |v| {
            v["record_per_request"] = json!(true);
            if !g3 {
                v["engine"]["rank"]
                    .as_object_mut()
                    .unwrap()
                    .remove("g3_offload");
            }
        });
        let (report, artifacts) = Replayer::new(input, ReplayEngineFactory::new())
            .unwrap()
            .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
            .unwrap();
        assert_eq!(report.request_counts.completed_requests, 3);
        let records = serde_json::to_value(&report.per_request).unwrap();
        let artifacts = serde_json::to_value(artifacts).unwrap();
        let uuid = |id: &str| {
            records
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["request_id"] == id)
                .unwrap()["uuid"]
                .clone()
        };
        let events = artifacts["host_offload_events"].as_array().unwrap();
        let mut mappings = events
            .iter()
            .filter(|r| {
                r["request_id"] == uuid("0") && r["event"]["event"] == "store_block_mappings"
            })
            .flat_map(|r| r["event"]["mappings"].as_array().unwrap())
            .map(|m| {
                (
                    m["logical_block_index"].as_u64().unwrap(),
                    m["block_hash"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        mappings.sort_unstable();
        assert_eq!(mappings.iter().map(|m| m.0).collect::<Vec<_>>(), vec![0, 1]);
        let expected = json!(mappings.iter().map(|m| m.1).collect::<Vec<_>>());
        for event in ["load_queued", "load_completed"] {
            let loads = events
                .iter()
                .filter(|r| r["request_id"] == uuid("2") && r["event"]["event"] == event)
                .collect::<Vec<_>>();
            assert_eq!(loads.len(), 1);
            assert_eq!(loads[0]["event"]["block_hashes"], expected);
        }
    }
}

#[test]
fn g3_zero_time_restore_finishes_without_future_arrival_in_both_scopes() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let input = g3_spec(
            scope,
            1,
            5,
            4,
            16,
            &[(0.0, 0, 32, 1), (100.0, 1000, 64, 1), (200.0, 0, 32, 1)],
        );
        let report = run_engine_replay(input).unwrap();
        assert_eq!(report.request_counts.completed_requests, 3);
        // Native decode timing has a 1ms floor even for fixed decode_ms=0.
        assert_eq!(report.throughput.duration_ms, 201.0);
        let g3 = report.g3_offload.unwrap();
        assert!(g3.read.completed_jobs > 0);
        assert!(g3.lookup_hits <= g3.lookup_probes);
        assert_eq!(g3.read.completed_jobs, g3.read.submitted_jobs);
    }
}

#[test]
fn g3_capacity_head_cannot_block_the_owner_of_reserved_g1() {
    // User-supplied independent five-request counterexample, preserved literally.
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let input = g3_spec(
            scope,
            1,
            6,
            7,
            4,
            &[
                (1.0, 100, 19, 0),
                (1.2, 0, 24, 0),
                (102.3, 0, 3, 2),
                (202.3, 100, 14, 1),
                (203.4, 0, 13, 2),
            ],
        );
        let input = g3_patch(input, |v| {
            let rank = &mut v["engine"]["rank"];
            rank["max_num_batched_tokens"] = json!(4);
            rank["kv_cache_bytes_per_token"] = json!(250000);
            rank["timing_model"]["prefill_ms"] = json!(10.0);
            rank["native_host_offload"]["h2d_bandwidth_gbps"] = json!(1.0);
            rank["g3_offload"]["num_g3_blocks"] = json!(19);
            rank["g3_offload"]["latency_to_first_byte_ms"] = json!(0.1);
            rank["g3_offload"]["read_bandwidth_gbps"] = json!(10.0);
            rank["g3_offload"]["shared_read_bandwidth_gbps"] = json!(10.0);
            rank["g3_offload"]["shared_write_bandwidth_gbps"] = json!(10.0);
        });
        let report = run_engine_replay(input).unwrap();
        assert_eq!(report.request_counts.completed_requests, 5);
        assert!(report.g3_offload.unwrap().read.completed_jobs > 0);
    }
}

#[test]
fn g3_replica_sharing_changes_only_reuse_and_is_repeatable() {
    let rows = [
        (0.0, 0, 8, 1),
        (100.0, 100, 8, 1),
        (200.0, 200, 8, 1),
        (300.0, 0, 8, 1),
    ];
    let mut results = Vec::new();
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let input = g3_spec(scope, 2, 3, 2, 4, &rows);
        let first = run_engine_replay(input.clone()).unwrap();
        let second = run_engine_replay(input).unwrap();
        assert_eq!(first.request_counts.completed_requests, 4);
        assert_eq!(first.g3_offload, second.g3_offload);
        let summaries = [&first, &second].map(|report| {
            let mut summary = serde_json::to_value(report).unwrap();
            // Only host execution speed is nondeterministic, not simulated timing.
            for field in [
                "wall_time_ms",
                "processed_tokens_per_s",
                "processed_output_tokens_per_s",
            ] {
                summary.as_object_mut().unwrap().remove(field);
            }
            summary
        });
        assert_eq!(summaries[0], summaries[1]);
        results.push(first.g3_offload.unwrap());
    }
    assert_eq!(results[0].cross_worker_read_blocks, 0);
    assert!(results[1].cross_worker_read_blocks > 0);
}

#[test]
fn g3_zero_output_compute_and_io_settle_at_last_arrival() {
    let input = g3_spec(
        G3Scope::ClusterShared,
        1,
        5,
        4,
        16,
        &[(0.0, 0, 32, 0), (100.0, 1000, 64, 0), (200.0, 0, 32, 0)],
    );
    let report = run_engine_replay(input).unwrap();
    assert_eq!(report.request_counts.completed_requests, 3);
    assert_eq!(report.throughput.duration_ms, 200.0);
    assert!(report.g3_offload.unwrap().read.completed_jobs > 0);
}

#[test]
fn g3_raw_replay_g3_without_g2_is_rejected_before_factory_strips_config() {
    let input = g3_spec(G3Scope::ClusterShared, 1, 5, 4, 16, &[(0.0, 0, 32, 1)]);
    let input = g3_patch(input, |v| {
        v["engine"]["rank"]
            .as_object_mut()
            .unwrap()
            .remove("native_host_offload");
    });
    let error = run_engine_replay(input).unwrap_err().to_string();
    assert!(
        error.contains("g3_offload requires native_host_offload"),
        "{error}"
    );
}

#[test]
fn g3_zero_duration_promotions_under_temporary_g2_pressure_fall_back_and_drain() {
    for scope in [G3Scope::WorkerLocal, G3Scope::ClusterShared] {
        let input = g3_spec(
            scope,
            1,
            2,
            2,
            4,
            &[
                (0.0, 0, 8, 1),
                (2002.0, 200, 1, 4),
                (2007.0, 100, 4, 1),
                (2009.0, 0, 8, 1),
            ],
        );
        let input = g3_patch(input, |v| {
            let rank = &mut v["engine"]["rank"];
            rank["max_num_seqs"] = json!(1);
            rank["kv_cache_bytes_per_token"] = json!(250000);
            rank["g3_offload"]["write_bandwidth_gbps"] = json!(0.001);
            rank["g3_offload"]["shared_write_bandwidth_gbps"] = json!(0.001);
        });
        // First prefix writes finish at2001ms. A generation-only full block
        // evicts one G1 block without a G2 prompt store. X evicts the other G1
        // block and starts a slow write at2008ms, pinning one G2 slot.
        // The final request's zero-time reads must not
        // alternate forever in the other slot or fabricate a protected hit.
        let report = run_engine_replay(input).unwrap();
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.throughput.duration_ms, 2010.0);
        let stats = report.g3_offload.unwrap();
        // Both reads start at the current lookup time2009; neither is backdated
        // to X's2008 boundary, and the A/B cycle terminates at this timestamp.
        assert_eq!(stats.read.completed_jobs, 2);
        assert_eq!(stats.read.completed_bytes, 2_000_000);
        assert_eq!(stats.pending_blocks, 0);
        assert_eq!(stats.write.submitted_jobs, stats.write.completed_jobs);
    }
}

#[test]
fn g3_post_lookup_touch_can_evict_a_later_prefix_block() {
    let input = g3_spec(
        G3Scope::WorkerLocal,
        1,
        2,
        2,
        4,
        &[
            (0.0, 0, 8, 1),
            (10.0, 200, 1, 4),
            (20.0, 100, 4, 1),
            (100.0, 0, 8, 1),
        ],
    );
    let input = g3_patch(input, |v| {
        let rank = &mut v["engine"]["rank"];
        rank["max_num_seqs"] = json!(1);
        rank["kv_cache_bytes_per_token"] = json!(250000);
        rank["g3_offload"]["latency_to_first_byte_ms"] = json!(2.0);
        rank["g3_offload"]["read_bandwidth_gbps"] = json!(1.0);
        rank["g3_offload"]["shared_read_bandwidth_gbps"] = json!(1.0);
    });
    let report = run_engine_replay(input).unwrap();
    assert_eq!(report.request_counts.completed_requests, 4);
    // At100ms A is missing. Reserving it precedes the full-prefix touch, so
    // it evicts the older B; reserving B then evicts X. Both blocks must be
    // read in one batch:2ms first-byte +2ms data, followed by1ms decode.
    assert_eq!(report.throughput.duration_ms, 105.0);
    let stats = report.g3_offload.unwrap();
    assert_eq!(stats.read.completed_jobs, 1);
    assert_eq!(stats.read.completed_bytes, 2_000_000);
    assert_eq!(stats.read.transfer_ms, 4.0);
}

#[test]
fn g3_idle_gap_does_not_consume_first_byte_or_transfer_time_before_arrival() {
    // A one-block restore has no later prefix block for lookup-touch ordering
    // to protect or evict. Both idle gaps must pay the same3ms read service.
    for arrival in [100.0, 1000.0] {
        let input = g3_spec(
            G3Scope::WorkerLocal,
            1,
            2,
            1,
            4,
            &[
                (0.0, 0, 4, 1),
                (10.0, 100, 4, 1),
                (20.0, 200, 4, 1),
                (arrival, 0, 4, 1),
            ],
        );
        let input = g3_patch(input, |v| {
            let rank = &mut v["engine"]["rank"];
            rank["max_num_seqs"] = json!(1);
            rank["kv_cache_bytes_per_token"] = json!(250000);
            rank["g3_offload"]["latency_to_first_byte_ms"] = json!(2.0);
            rank["g3_offload"]["read_bandwidth_gbps"] = json!(1.0);
            rank["g3_offload"]["shared_read_bandwidth_gbps"] = json!(1.0);
        });
        let report = run_engine_replay(input).unwrap();
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.throughput.duration_ms, arrival + 4.0);
        let stats = report.g3_offload.unwrap();
        assert_eq!(stats.read.completed_jobs, 1);
        assert_eq!(stats.read.completed_bytes, 1_000_000);
        assert_eq!(stats.read.transfer_ms, 3.0);
    }
}

#[test]
fn g3_summary_serializes_g3_counters_only_when_enabled() {
    let input = g3_spec(G3Scope::ClusterShared, 1, 5, 4, 16, &[(0.0, 0, 32, 1)]);
    let report = run_engine_replay(input.clone()).unwrap();
    let summary = serde_json::to_value(&report).unwrap();
    assert_eq!(
        summary["g3_offload"],
        serde_json::to_value(&report.g3_offload).unwrap()
    );
    assert!(
        summary["g3_offload"]["write"]["completed_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    let input = g3_patch(input, |v| {
        v["engine"]["rank"]
            .as_object_mut()
            .unwrap()
            .remove("g3_offload");
    });
    let report = run_engine_replay(input).unwrap();
    assert!(report.g3_offload.is_none());
    assert!(
        serde_json::to_value(report)
            .unwrap()
            .get("g3_offload")
            .is_none()
    );
}

#[test]
fn agg_settled_steps_resume_without_changing_the_result() {
    let pending = VecDeque::from([request(1, 0.0), request(2, 3.0)]);
    let (continuous, _) = runtime(pending.clone()).run().unwrap();

    let mut stepped = runtime(pending);
    let mut settled_boundaries = 0;
    loop {
        match stepped.step().unwrap() {
            ReplayStepOutcome::Settled { .. } => settled_boundaries += 1,
            ReplayStepOutcome::Complete => break,
            ReplayStepOutcome::TimeLimitReached { .. } => panic!("unexpected time limit"),
        }
    }
    let (resumed, _) = stepped.run().unwrap();

    assert!(settled_boundaries >= 2);
    assert_eq!(
        serde_json::to_value(continuous.finish()).unwrap(),
        serde_json::to_value(resumed.finish()).unwrap()
    );
}
