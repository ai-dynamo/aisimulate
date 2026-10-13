// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-level KVCR: a hinted request pulls a prefix from a peer rank's G2.

use aisimulate_core::engine::KvcrPullCounts;
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, PoolRoundRobinPlacement,
    ReplayCaptureOptions, ReplayComposition, ReplayDeterminism, ReplayEngineFactory, ReplayReport,
    ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec, Replayer,
    RoundRobinComposition, WorkerTopology,
};
use serde_json::{Value, json};

/// 9-token prompts: two complete 4-token blocks of 1 MB can be pulled.
const PROMPT: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
/// Shares no block with `PROMPT`.
const OTHER: [u32; 9] = [21, 22, 23, 24, 25, 26, 27, 28, 29];
/// `PROMPT`'s two blocks plus a third.
const LONG: [u32; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

fn rank(host_offload: Value) -> Value {
    json!({
        "num_gpu_blocks": 16, "block_size": 4, "max_num_seqs": 4,
        "max_num_batched_tokens": 64, "kv_cache_bytes_per_token": 250_000,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        "native_host_offload": host_offload,
    })
}

fn kvcr_rank(kvcr: Value) -> Value {
    let mut rank = rank(json!({"num_host_blocks": 8}));
    rank["kvcr"] = kvcr;
    rank
}

fn request(id: &str, arrival_ms: f64, tokens: &[u32]) -> Value {
    json!({
        "id": id, "arrival_time_ms": arrival_ms, "input_tokens": tokens.len(),
        "input_token_ids": tokens, "output_tokens": 1,
    })
}

/// A request whose hint names DP rank 0 of `source_worker`.
fn hinted(id: &str, arrival_ms: f64, tokens: &[u32], source_worker: usize) -> Value {
    let mut request = request(id, arrival_ms, tokens);
    request["kv_hints"] = json!({"source_worker": source_worker, "source_dp_rank": 0});
    request
}

/// r0 computes PROMPT on DP rank 0; r1 arrives later on DP rank 1, hinted at rank 0.
fn seed_then_pull() -> Vec<Value> {
    let mut r0 = request("r0", 0.0, &PROMPT);
    r0["dp_rank"] = json!(0);
    let mut r1 = hinted("r1", 100.0, &PROMPT, 0);
    r1["dp_rank"] = json!(1);
    vec![r0, r1]
}

fn run(topology: Value, engine: Value, requests: Vec<Value>) -> anyhow::Result<ReplayReport> {
    run_with(RoundRobinComposition, topology, engine, requests)
}

fn run_with(
    composition: impl ReplayComposition,
    topology: Value,
    engine: Value,
    requests: Vec<Value>,
) -> anyhow::Result<ReplayReport> {
    let spec: ReplaySpec = serde_json::from_value(json!({
        "version": 1, "topology": topology, "engine": engine,
        "requests": requests, "record_per_request": true,
    }))?;
    Ok(
        Replayer::with_composition(spec, ReplayEngineFactory::new(), composition)?
            .with_capture_options(ReplayCaptureOptions {
                determinism: ReplayDeterminism::CanonicalV1,
                ..ReplayCaptureOptions::default()
            })
            .run()?,
    )
}

/// Like `run`, but each worker tokenizes a request for 5 ms before its engine sees it.
fn run_tokenized(topology: Value, engine: Value, requests: Vec<Value>) -> ReplayReport {
    let spec: ReplaySpec = serde_json::from_value(json!({
        "version": 1, "topology": topology, "engine": engine,
        "requests": requests, "record_per_request": true,
    }))
    .unwrap();
    Replayer::with_composition(spec, ReplayEngineFactory::new(), RoundRobinComposition)
        .unwrap()
        .with_tokenization(1, 5.0, 0.0)
        .unwrap()
        .run()
        .unwrap()
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
fn a_hinted_request_pulls_its_prefix_from_a_peer_rank() {
    let engine = json!({
        "dp_size": 2,
        "rank": kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0})),
    });
    let report = run(aggregated(1), engine, seed_then_pull()).unwrap();
    assert_eq!(host_reuse(&report), vec![Some(0), Some(8)]);
    let kvcr = report.kvcr.expect("kvcr section");
    assert_eq!(
        kvcr.pulls,
        KvcrPullCounts {
            success: 1,
            partial: 0
        }
    );
    assert_eq!((kvcr.blocks, kvcr.loaded_blocks), (2, 2));
    // 1 ms latency + 2 blocks of 1 MB at 1 GB/s.
    assert_eq!(kvcr.remote_deliver_ms, 3.0);
}

#[test]
fn a_pull_moves_the_kv_of_every_tensor_parallel_gpu() {
    let engine = json!({
        "dp_size": 2, "tensor_parallel_size": 4,
        "rank": kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0})),
    });
    let report = run(aggregated(1), engine, seed_then_pull()).unwrap();
    assert_eq!(host_reuse(&report), vec![Some(0), Some(8)]);
    let kvcr = report.kvcr.expect("kvcr section");
    // Each G2 block holds one GPU's 1 MB; a pull moves all 4 GPUs' shares.
    assert_eq!((kvcr.blocks, kvcr.bytes), (2, 2 * 1_000_000 * 4));
    // 1 ms latency + 8 MB at 1 GB/s.
    assert_eq!(kvcr.remote_deliver_ms, 9.0);
}

#[test]
fn a_zero_time_pull_finishes_without_a_future_arrival() {
    let mut rank = kvcr_rank(json!({"latency_to_first_byte_ms": 0.0, "bandwidth_gbps": 0.0}));
    rank["timing_model"] = json!({"type": "fixed", "prefill_ms": 0.0, "decode_ms": 0.0});
    rank["native_host_offload"] =
        json!({"num_host_blocks": 8, "d2h_bandwidth_gbps": 0.0, "h2d_bandwidth_gbps": 0.0});
    let report = run(
        aggregated(1),
        json!({"dp_size": 2, "rank": rank}),
        seed_then_pull(),
    )
    .unwrap();
    // r1 is the last arrival. Its pull lands when the zero-duration pass that
    // staged it completes; that pass shows no effect, so the worker waits in
    // the deferred-ready set. The arrival changed state at 100 ms, so Replay
    // drains that instant again, re-drives the worker and r1 loads the pull.
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.throughput.duration_ms, 100.0);
    assert_eq!(host_reuse(&report), vec![Some(0), Some(8)]);
    assert_eq!(report.kvcr.unwrap().pulls.success, 1);
}

#[test]
fn a_busy_source_delays_the_pull_until_its_pass_ends() {
    let mut rank = kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0}));
    rank["timing_model"] = json!({"type": "fixed", "prefill_ms": 50.0, "decode_ms": 1.0});
    // Round-robin alternates workers: r0 seeds worker 0 and r2 keeps it busy
    // over [99, 149); x runs on worker 1, where r1 lands at 100.
    let requests = vec![
        request("r0", 0.0, &PROMPT),
        request("x", 1.0, &OTHER),
        request("r2", 99.0, &OTHER),
        hinted("r1", 100.0, &PROMPT, 0),
    ];
    let report = run(aggregated(2), json!({"rank": rank}), requests).unwrap();
    let kvcr = report.kvcr.unwrap();
    // The pull waits 149 - 100 ms, then 1 ms latency and 2 MB at 1 GB/s.
    assert_eq!(kvcr.source_wait_ms, 49.0);
    assert_eq!(kvcr.remote_deliver_ms, 52.0);
}

#[test]
fn a_source_pass_starting_at_the_same_instant_delays_the_pull_whatever_the_worker_order() {
    let mut rank = kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0}));
    rank["timing_model"] = json!({"type": "fixed", "prefill_ms": 50.0, "decode_ms": 1.0});
    // Round-robin alternates workers 0 and 1. The first two requests seed
    // PROMPT on the source and OTHER on the target. At 100, the hinted r1
    // reaches the target and r2 (a prompt new to the source) reaches the
    // source, which prefills it over [100, 150). Both workers start a pass at
    // 100, whichever has the lower id.
    let cases = [
        // Target worker 0 (scanned first), source worker 1.
        [
            request("x", 0.0, &OTHER),
            request("r0", 1.0, &PROMPT),
            hinted("r1", 100.0, &PROMPT, 1),
            request("r2", 100.0, &OTHER),
        ],
        // Source worker 0 (scanned first), target worker 1.
        [
            request("r0", 0.0, &PROMPT),
            request("x", 1.0, &OTHER),
            request("r2", 100.0, &OTHER),
            hinted("r1", 100.0, &PROMPT, 0),
        ],
    ];
    for requests in cases {
        let report = run(
            aggregated(2),
            json!({"rank": rank.clone()}),
            requests.into(),
        )
        .unwrap();
        let kvcr = report.kvcr.unwrap();
        assert_eq!(kvcr.pulls.success, 1);
        // The pull waits 150 - 100 ms, then 1 ms latency and 2 MB at 1 GB/s.
        assert_eq!(kvcr.source_wait_ms, 50.0);
        assert_eq!(kvcr.remote_deliver_ms, 53.0);
    }
}

#[test]
fn a_dp_sibling_pull_waits_for_the_shared_pass() {
    let mut rank = kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0}));
    rank["timing_model"] = json!({"type": "fixed", "prefill_ms": 50.0, "decode_ms": 1.0});
    // r2 arrives on DP rank 0 with r1, so the worker's pass that stages r1's
    // pull from rank 0 runs r2's prefill over [100, 150).
    let mut requests = seed_then_pull();
    let mut r2 = request("r2", 100.0, &OTHER);
    r2["dp_rank"] = json!(0);
    requests.push(r2);
    let report = run(aggregated(1), json!({"dp_size": 2, "rank": rank}), requests).unwrap();
    let kvcr = report.kvcr.unwrap();
    // The pull waits 150 - 100 ms, then 1 ms latency and 2 MB at 1 GB/s.
    assert_eq!(kvcr.source_wait_ms, 50.0);
    assert_eq!(kvcr.remote_deliver_ms, 53.0);
}

#[test]
fn two_targets_share_one_source_bandwidth() {
    // Round-robin: r0 seeds worker 0, then r1 and r2 land on workers 1 and 2 at once.
    let requests = vec![
        request("r0", 0.0, &PROMPT),
        hinted("r1", 100.0, &PROMPT, 0),
        hinted("r2", 100.0, &PROMPT, 0),
    ];
    // Each pull waits 1 ms. The two 2 MB pulls then share worker 0's 1 GB/s
    // for 4 ms; an unlimited source moves them at once.
    for (gbps, total_ms) in [(1.0, 2.0 * (1.0 + 4.0)), (0.0, 2.0 * 1.0)] {
        let timing = json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": gbps});
        let engine = json!({"rank": kvcr_rank(timing)});
        let report = run(aggregated(3), engine, requests.clone()).unwrap();
        let kvcr = report.kvcr.unwrap();
        assert_eq!(kvcr.pulls.success, 2);
        assert_eq!(kvcr.remote_deliver_ms, total_ms);
    }
}

#[test]
fn one_pull_serves_two_requests_for_the_same_prefix() {
    let mut requests = seed_then_pull();
    let mut r2 = requests[1].clone();
    r2["id"] = json!("r2");
    requests.push(r2);
    let engine = json!({"dp_size": 2, "rank": kvcr_rank(json!({}))});
    let report = run(aggregated(1), engine, requests).unwrap();
    // r2 waits on r1's pull and then on r1's load of it, so it reuses the
    // same 8 tokens from G1 rather than loading them again.
    assert_eq!(host_reuse(&report), vec![Some(0), Some(8), Some(0)]);
    let r2 = report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some("r2"))
        .unwrap();
    assert_eq!(r2.first_admission_g1_reused_input_tokens, Some(8));
    let kvcr = report.kvcr.unwrap();
    assert_eq!((kvcr.pulls.success, kvcr.loaded_blocks), (1, 2));
}

/// Round-robin placement whose scaling drains worker 1 at 102 ms, while it
/// sources a pull that lands at 103 ms (scale-in drains the newest worker).
struct DrainSourceMidPull;

impl ReplayComposition for DrainSourceMidPull {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        _prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        _decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        unreachable!("this test replays an aggregated deployment")
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(Some(Box::new(DrainSourceMidPull)))
    }
}

impl ReplayScalingPolicy for DrainSourceMidPull {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(102.0)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        // Worker 1 still drains while its pull is in flight and is gone after it lands.
        let (draining, next_tick_ms) = match snapshot.tick_ordinal {
            0 => (vec![], Some(102.5)),
            1 => (vec![1], Some(104.0)),
            _ => (vec![], None),
        };
        assert_eq!(
            snapshot.draining_decode_ids, draining,
            "draining workers at {} ms",
            snapshot.now_ms
        );
        Ok(ReplayScalingDecision {
            target_decode: Some(1),
            next_tick_ms,
            ..ReplayScalingDecision::default()
        })
    }
}

#[test]
fn a_draining_source_is_removed_after_its_pulls_land() {
    // Round-robin: x runs on worker 0 and r1 seeds LONG on worker 1. r2 on
    // worker 0 pulls PROMPT's two blocks over [100, 103). r3 arrives after
    // worker 1 left; worker 0 lacks LONG's third block, but the hint is ignored.
    let engine = json!({
        "rank": kvcr_rank(json!({"latency_to_first_byte_ms": 1.0, "bandwidth_gbps": 1.0})),
    });
    let requests = vec![
        request("x", 0.0, &OTHER),
        request("r1", 1.0, &LONG),
        hinted("r2", 100.0, &PROMPT, 1),
        hinted("r3", 200.0, &LONG, 1),
    ];
    let report = run_with(DrainSourceMidPull, aggregated(2), engine, requests).unwrap();
    assert_eq!(report.request_counts.completed_requests, 4);
    assert_eq!(
        host_reuse(&report),
        vec![Some(0), Some(8), Some(0), Some(0)]
    );
    assert_eq!(report.kvcr.unwrap().pulls.success, 1);
}

#[test]
fn a_hint_survives_tokenization_before_dispatch() {
    // Round-robin places r0 on worker 0 and the hinted r1 on worker 1, in both
    // topologies; the hint must reach the engine after tokenization.
    let requests = vec![request("r0", 0.0, &PROMPT), hinted("r1", 100.0, &PROMPT, 0)];
    let aggregated_engine = json!({"rank": kvcr_rank(json!({}))});
    let pd_engine = json!({
        "rank": rank(Value::Null),
        "prefill": {"rank": kvcr_rank(json!({}))},
        "decode": {"rank": rank(Value::Null)},
    });
    for (topology, engine) in [
        (aggregated(2), aggregated_engine),
        (disaggregated(2), pd_engine),
    ] {
        let report = run_tokenized(topology, engine, requests.clone());
        assert_eq!(host_reuse(&report), vec![Some(0), Some(8)]);
        assert_eq!(report.kvcr.unwrap().pulls.success, 1);
    }
}

#[test]
fn prefill_ranks_pull_from_each_other_in_pd() {
    let pd = |decode: Value| {
        json!({
            "rank": rank(Value::Null),
            "prefill": {"rank": kvcr_rank(json!({}))},
            "decode": {"rank": decode},
        })
    };
    // Round-robin: r0 seeds prefill worker 0 and r1 lands on prefill worker 1.
    let requests = vec![request("r0", 0.0, &PROMPT), hinted("r1", 100.0, &PROMPT, 0)];
    let report = run(disaggregated(2), pd(rank(Value::Null)), requests.clone()).unwrap();
    assert_eq!(host_reuse(&report), vec![Some(0), Some(8)]);
    assert_eq!(report.kvcr.unwrap().pulls.success, 1);

    let error = run(disaggregated(2), pd(kvcr_rank(json!({}))), requests).unwrap_err();
    let error = format!("{error:#}");
    assert!(
        error.contains("kvcr supports aggregated and prefill workers only"),
        "{error}"
    );
}
