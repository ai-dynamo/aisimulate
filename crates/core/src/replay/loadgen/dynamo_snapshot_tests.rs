// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-authored Dynamo records distinguish source intervals from the existing
//! completion-relative execution clocks.

use super::{
    AgenticSnapshotOptions, DynamoRequestTrace, PreparedAgenticSnapshots, ValidatedAgenticGraph,
    WorkloadDriver,
};

fn graph() -> ValidatedAgenticGraph {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("requests.jsonl");
    let records = ["a", "b", "c"]
        .into_iter()
        .enumerate()
        .map(|(index, id)| {
            let start = 10_000 + index as u64 * 1000;
            let mut record = serde_json::json!({
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": start + 100,
                "agent_context": {"session_id": "conversation"},
                "request": {
                    "request_id": id, "model": "model", "output_tokens": 2,
                    "request_received_ms": start, "total_time_ms": 100,
                    "replay": {
                        "trace_block_size": 64, "input_length": 128,
                        "input_sequence_hashes": [10, 20]
                    }
                }
            });
            if id == "b" {
                // Missing duration retains the existing event-end fallback.
                record["request"]
                    .as_object_mut()
                    .unwrap()
                    .remove("total_time_ms");
            }
            serde_json::to_string(&record).unwrap()
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&path, records + "\n").unwrap();
    let DynamoRequestTrace::Agentic(graph) =
        DynamoRequestTrace::from_request_trace_files(&[path], Some(64)).unwrap()
    else {
        panic!("contextual Dynamo records must lower to an agentic graph");
    };
    graph
}

#[test]
fn dynamo_snapshot_uses_original_intervals_for_history_and_frontier() {
    let graph = graph();
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let sampled = &prepared.snapshots()[0];
    assert!((500.0..1500.0).contains(&sampled.t_star_ms));
    assert_eq!(sampled.recorded_start_ms, 0.0);
    assert_eq!(sampled.recorded_last_start_ms, 2000.0);
    assert_eq!(
        sampled
            .requests
            .iter()
            .map(|request| (
                request.source_request_id.as_str(),
                request.recorded_start_ms,
                request.recorded_end_ms
            ))
            .collect::<Vec<_>>(),
        [
            ("a", 0.0, Some(100.0)),
            ("b", 1000.0, Some(1100.0)),
            ("c", 2000.0, Some(2100.0))
        ]
    );
    assert!(sampled.requests.iter().any(|request| request.historical));
    assert!(sampled.requests.iter().any(|request| !request.historical));
    assert_eq!(sampled.primers.len(), 1);

    let exact = prepared.context().prepare_play(0, 0, Some(1000.0)).unwrap();
    let requests = &exact.evidence().requests;
    assert_eq!(
        requests
            .iter()
            .map(|request| request.historical)
            .collect::<Vec<_>>(),
        [true, false, false]
    );
    assert_eq!(requests[1].remaining_delay_ms, 0.0);
    assert!(requests[1].pending_dependencies.is_empty());
    assert_eq!(requests[2].remaining_delay_ms, 0.0);
    assert_eq!(
        requests[2].pending_dependencies,
        [exact.identity("b").unwrap().request_id]
    );
    assert_eq!(exact.evidence().primers[0].source_request_id, "a");
    assert_eq!(exact.evidence().primers[0].input_length, 128);

    // A request already started at the cut remains history even while its
    // recorded service interval is still active; only the suffix is executed.
    let active = prepared.context().prepare_play(0, 0, Some(1050.0)).unwrap();
    assert!(active.evidence().requests[1].historical);
    assert_eq!(active.evidence().requests[1].recorded_end_ms, Some(1100.0));
    assert_eq!(active.evidence().primers[0].source_request_id, "b");
    let mut driver = WorkloadDriver::new_agentic_snapshots(
        PreparedAgenticSnapshots::from_plays(vec![active]).unwrap(),
        64,
        true,
        2.0,
    )
    .unwrap();
    assert_eq!(driver.total_turns(), 1);
    assert_eq!(driver.next_ready_time_ms(), Some(475.0));
    assert!(driver.pop_ready(474.0, 1).is_empty());
    let ready = driver.pop_ready(475.0, 1).pop().unwrap();
    assert_eq!(ready.authored_request_id.as_deref(), Some("c"));
    driver.on_complete(ready.request_uuid, 480.0).unwrap();
    assert!(driver.is_drained());
}

#[test]
fn dynamo_snapshot_live_dependencies_use_actual_completion_after_the_cut() {
    let prepared = graph()
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 0 })
        .unwrap();
    for cut_ms in [600.0, 1000.0] {
        for speedup in [1.0, 2.0] {
            let snapshot = prepared.context().prepare_play(0, 0, Some(cut_ms)).unwrap();
            assert_eq!(snapshot.evidence().primers[0].source_request_id, "a");
            let mut driver = WorkloadDriver::new_agentic_snapshots(
                PreparedAgenticSnapshots::from_plays(vec![snapshot]).unwrap(),
                64,
                true,
                speedup,
            )
            .unwrap();
            assert_eq!(driver.total_turns(), 2);
            // The historical a→b edge retains its recorded deadline. The live
            // b→c edge retains 900ms of think time after actual b completion,
            // even when replay service is faster than the recorded 100ms.
            let b_at_ms = (1000.0 - cut_ms) / speedup;
            let c_at_ms = b_at_ms + 10.0 + 900.0 / speedup;
            for (id, at_ms) in [("b", b_at_ms), ("c", c_at_ms)] {
                assert_eq!(driver.next_ready_time_ms(), Some(at_ms));
                assert!(driver.pop_ready(at_ms - 1.0, 1).is_empty());
                let mut ready = driver.pop_ready(at_ms, usize::MAX);
                assert_eq!(ready.len(), 1);
                let ready = ready.pop().unwrap();
                assert_eq!(ready.authored_request_id.as_deref(), Some(id));
                driver
                    .on_complete(ready.request_uuid, at_ms + 10.0)
                    .unwrap();
            }
            assert!(driver.is_drained());
        }
    }
}

#[test]
fn private_dynamo_source_intervals_do_not_change_serialized_nodes_or_graph_digest() {
    let graph = graph();
    let mut without_intervals = graph.clone();
    assert!(
        graph
            .nodes()
            .iter()
            .all(|node| node.recorded_interval_ms.is_some())
    );
    for node in &mut without_intervals.nodes {
        node.recorded_interval_ms = None;
    }
    assert_eq!(
        serde_json::to_value(graph.nodes()).unwrap(),
        serde_json::to_value(without_intervals.nodes()).unwrap()
    );
    assert_eq!(graph.identity(), without_intervals.identity());
    // This public no-op also recomputes the canonical digest, avoiding a check
    // that merely compares the cached digest strings copied by clone().
    let refreshed = graph.speed_up_timing(1.0).unwrap();
    let without_intervals = without_intervals.speed_up_timing(1.0).unwrap();
    assert_eq!(refreshed.identity(), without_intervals.identity());
}

#[test]
fn dynamo_legacy_driver_retains_completion_relative_delays() {
    let graph = graph();
    assert!(graph.nodes().iter().all(|node| node.not_before_ms() == 0.0));
    assert!(
        graph
            .nodes()
            .iter()
            .all(|node| node.recorded_api_time_ms().is_none())
    );
    assert_eq!(graph.nodes()[1].dependencies()[0].delay_ms, 900.0);
    assert_eq!(graph.nodes()[2].dependencies()[0].delay_ms, 900.0);
    let mut driver = WorkloadDriver::new_agentic_trace(graph, 64).unwrap();
    for (id, at_ms, completion_ms) in [("a", 0.0, 10.0), ("b", 910.0, 920.0), ("c", 1820.0, 1830.0)]
    {
        assert_eq!(driver.next_ready_time_ms(), Some(at_ms));
        let mut ready = driver.pop_ready(at_ms, usize::MAX);
        assert_eq!(ready.len(), 1);
        let ready = ready.pop().unwrap();
        assert_eq!(ready.authored_request_id.as_deref(), Some(id));
        driver
            .on_complete(ready.request_uuid, completion_ms)
            .unwrap();
    }
    assert!(driver.is_drained());
}
