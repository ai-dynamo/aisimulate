// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-authored request-boundary fixtures. These test logical snapshot state;
//! the sibling replay suite qualifies physical cache reuse in native engines.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

fn row(id: &str, conversation: &str, start: f64, duration: Option<f64>) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: id.into(),
        play_id: "play".into(),
        session_id: conversation.into(),
        model: "model".into(),
        input_length: Some(128),
        output_length: Some(3),
        hash_ids: Some(vec![10, 20]),
        not_before_ms: start,
        recorded_api_time_ms: duration,
        ..Default::default()
    }
}

fn edge(
    source: &str,
    relation: AgenticDependencyRelation,
    trigger: AgenticDependencyTrigger,
    delay: f64,
) -> AgenticDependency {
    AgenticDependency {
        request_id: source.into(),
        relation,
        trigger,
        delay_ms: delay,
    }
}

fn graph(rows: Vec<AgenticMooncakeRow>) -> ValidatedAgenticGraph {
    ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 64,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored-test".into(),
                digest: "snapshot-fixture-v1".into(),
            },
        },
        rows,
    )
    .unwrap()
}

fn boundary_graph() -> ValidatedAgenticGraph {
    use AgenticDependencyRelation::{Sequence, Spawn};
    use AgenticDependencyTrigger::{Completion, Dispatch};
    let root = row("root", "main", 0.0, Some(120.0));
    let mut child = row("child", "worker", 20.0, Some(80.0));
    child.dependencies = vec![edge("root", Spawn, Dispatch, 20.0)];
    let mut background = row("background", "finished-conversation", 80.0, Some(500.0));
    background.dependencies = vec![edge("root", Spawn, Dispatch, 80.0)];
    let mut boundary = row("at-cut", "main", 100.0, Some(40.0));
    boundary.dependencies = vec![edge("root", Sequence, Completion, 10.0)];
    let mut child_next = row("child-next", "worker", 140.0, Some(20.0));
    child_next.dependencies = vec![edge("child", Sequence, Completion, 10.0)];
    let mut future = row("future", "main", 160.0, None);
    future.dependencies = vec![edge("at-cut", Sequence, Completion, 20.0)];
    graph(vec![root, child, background, boundary, child_next, future])
}

fn request<'a>(play: &'a AgenticPlaySnapshot, source: &str) -> &'a AgenticSnapshotRequest {
    play.evidence()
        .requests
        .iter()
        .find(|request| request.source_request_id == source)
        .unwrap()
}

fn prepare(graph: &ValidatedAgenticGraph, cut: f64) -> AgenticPlaySnapshot {
    graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 42 })
        .unwrap()
        .context()
        .prepare_play(0, 0, Some(cut))
        .unwrap()
}

fn drain(mut driver: WorkloadDriver) -> BTreeMap<String, ReadyTurn> {
    let mut requests = BTreeMap::new();
    while !driver.is_drained() {
        let now = driver
            .next_ready_time_ms()
            .expect("fixture must have a future wakeup");
        let ready = driver.pop_ready(now, usize::MAX);
        assert!(!ready.is_empty());
        for ready in ready {
            driver.on_complete(ready.request_uuid, now).unwrap();
            requests.insert(ready.authored_request_id.clone().unwrap(), ready);
        }
    }
    requests
}

#[test]
fn exact_cut_preserves_active_history_and_original_identity_with_residual_timers() {
    let graph = boundary_graph();
    let play = prepare(&graph, 100.0);
    assert_eq!(play.evidence().graph, graph.identity());
    assert_eq!(play.evidence().t_star_ms, 100.0);
    assert!(request(&play, "root").historical);
    assert_eq!(request(&play, "root").recorded_end_ms, Some(120.0));
    assert!(request(&play, "background").historical);
    assert_eq!(request(&play, "background").recorded_end_ms, Some(580.0));
    assert!(!request(&play, "at-cut").historical);
    assert_eq!(request(&play, "at-cut").remaining_delay_ms, 30.0);
    assert!(request(&play, "at-cut").pending_dependencies.is_empty());
    assert_eq!(request(&play, "child-next").remaining_delay_ms, 40.0);
    assert_eq!(request(&play, "future").remaining_delay_ms, 60.0);
    assert_eq!(
        request(&play, "future").pending_dependencies,
        vec![play.identity("at-cut").unwrap().request_id]
    );
    let mut primers = play
        .evidence()
        .primers
        .iter()
        .map(|primer| primer.source_request_id.as_str())
        .collect::<Vec<_>>();
    primers.sort_unstable();
    assert_eq!(primers, ["child", "root"]);
    assert!(
        play.evidence()
            .primers
            .iter()
            .all(|primer| primer.input_length == 128)
    );
    let root_id = play.identity("root").unwrap().request_id;
    assert_eq!(
        play.identity("child").unwrap().parent_id.as_deref(),
        Some(root_id.as_str())
    );
    assert_eq!(
        play.identity("future").unwrap().root_id.as_deref(),
        Some(root_id.as_str())
    );
    let turn_zero = play.context().prepare_play(0, 0, Some(0.0)).unwrap();
    assert!(Arc::ptr_eq(play.context(), turn_zero.context()));
    for node in graph.nodes() {
        assert_eq!(
            play.identity(node.request_id()).unwrap(),
            turn_zero.identity(node.request_id()).unwrap()
        );
    }
}

#[test]
fn snapshot_driver_scales_timers_and_waits_for_actual_live_completion() {
    let graph = boundary_graph();
    let play = prepare(&graph, 100.0);
    let expected_evidence = play.evidence().clone();
    let mut driver = WorkloadDriver::new_agentic_snapshots(
        PreparedAgenticSnapshots::from_plays(vec![play.clone()]).unwrap(),
        48,
        true,
        2.0,
    )
    .unwrap();
    assert_eq!(driver.total_turns(), 3);
    assert_eq!(
        driver.agentic_snapshot_evidence(),
        Some(std::slice::from_ref(&expected_evidence))
    );
    assert_eq!(driver.next_ready_time_ms(), Some(15.0));
    assert!(driver.pop_ready(14.0, usize::MAX).is_empty());
    let mut ready = driver.pop_ready_compact(15.0, usize::MAX);
    assert_eq!(ready.len(), 1);
    let first = ready.pop().unwrap();
    assert_eq!(first.authored_request_id.as_deref(), Some("at-cut"));
    assert!(first.request.materialized_tokens().is_none());
    assert_eq!(
        first
            .request
            .metadata()
            .replay_context
            .as_ref()
            .unwrap()
            .agentic,
        Some(play.identity("at-cut").unwrap())
    );
    assert_eq!(
        first.replay_hashes,
        Some(play.replay_hashes("at-cut", 128, 48).unwrap())
    );
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));
    let child = driver.pop_ready(20.0, usize::MAX).pop().unwrap();
    assert_eq!(child.authored_request_id.as_deref(), Some("child-next"));
    driver.on_complete(child.request_uuid, 25.0).unwrap();
    // The recorded completion was 140 ms, but the live executor completes at
    // 35 ms: its 20 / 2 ms edge delay releases the dependent at 45 ms.
    driver.on_complete(first.request_uuid, 35.0).unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(45.0));
    assert!(driver.pop_ready(44.0, usize::MAX).is_empty());
    let future = driver.pop_ready(45.0, usize::MAX).pop().unwrap();
    assert_eq!(future.authored_request_id.as_deref(), Some("future"));
    driver.on_complete(future.request_uuid, 50.0).unwrap();
    assert!(driver.is_drained());
    let transcript = driver.agentic_lifecycle_transcript().unwrap();
    let dispatched = transcript
        .events
        .iter()
        .filter(|event| event.event == AgenticLifecycleEventKind::Dispatch)
        .collect::<Vec<_>>();
    assert_eq!(dispatched.len(), 3);
    assert_eq!(
        dispatched
            .iter()
            .map(|event| event.at_ms)
            .collect::<Vec<_>>(),
        [15.0, 20.0, 45.0]
    );
}

#[test]
fn primer_uses_chronological_predecessor_not_lexicographic_request_order() {
    let mut rows = Vec::new();
    for (index, start, previous) in [
        (0, 0.0, None),
        (2, 20.0, Some(0)),
        (10, 100.0, Some(2)),
        (11, 110.0, Some(10)),
    ] {
        let mut next = row(&format!("outer:{index}"), "main", start, Some(1.0));
        if let Some(previous) = previous {
            next.dependencies.push(edge(
                &format!("outer:{previous}"),
                AgenticDependencyRelation::Sequence,
                AgenticDependencyTrigger::Completion,
                0.0,
            ));
        }
        rows.push(next);
    }
    let graph = graph(rows);
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(AgenticNode::request_id)
            .collect::<Vec<_>>(),
        ["outer:0", "outer:10", "outer:11", "outer:2"]
    );
    let play = prepare(&graph, 105.0);
    assert_eq!(play.evidence().primers.len(), 1);
    assert_eq!(play.evidence().primers[0].source_request_id, "outer:10");
    assert_eq!(
        play.evidence().primers[0].request_id,
        play.identity("outer:10").unwrap().request_id
    );
}

#[test]
fn equal_timestamp_primer_follows_sequence_dependency_order() {
    let root = row("outer:0", "main", 0.0, Some(0.0));
    let mut predecessor = row("outer:2", "main", 10.0, Some(0.0));
    predecessor.dependencies.push(edge(
        "outer:0",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let mut latest = row("outer:10", "main", 10.0, Some(0.0));
    latest.dependencies.push(edge(
        "outer:2",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let mut future = row("outer:11", "main", 20.0, Some(0.0));
    future.dependencies.push(edge(
        "outer:10",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let graph = graph(vec![root, predecessor, latest, future]);
    let play = prepare(&graph, 15.0);
    assert_eq!(play.evidence().primers.len(), 1);
    assert_eq!(play.evidence().primers[0].source_request_id, "outer:10");
}

fn cycle_graph() -> ValidatedAgenticGraph {
    let mut rows = Vec::new();
    for play in ["a", "b"] {
        for (index, start) in [(0, 0.0), (1, 100.0)] {
            let mut request = row(&format!("{play}{index}"), play, start, Some(1.0));
            request.play_id = play.into();
            request.input_length = Some(64);
            request.hash_ids = Some(vec![10 + index]);
            rows.push(request);
        }
    }
    graph(rows)
}

#[test]
fn seeded_snapshots_are_repeatable_and_cycle_lanes_without_completion_order() {
    let graph = cycle_graph();
    let first = graph
        .prepare_snapshots(3, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let second = graph
        .prepare_snapshots(3, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    assert_eq!(
        serde_json::to_vec(first.snapshots()).unwrap(),
        serde_json::to_vec(second.snapshots()).unwrap()
    );
    assert_eq!(
        first
            .snapshots()
            .iter()
            .map(|snapshot| snapshot.source_play_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "a"]
    );
    assert!(
        first
            .snapshots()
            .iter()
            .all(|snapshot| (25.0..75.0).contains(&snapshot.t_star_ms))
    );
    let changed = graph
        .prepare_snapshots(3, AgenticSnapshotOptions { seed: 43 })
        .unwrap();
    assert_ne!(
        first.snapshots()[0].t_star_ms,
        changed.snapshots()[0].t_star_ms
    );
    let context = first.context();
    let late_first = context.prepare_play(2, 1, Some(0.0)).unwrap();
    let early_second = context.prepare_play(0, 1, Some(0.0)).unwrap();
    assert_eq!(
        late_first.evidence(),
        context.prepare_play(2, 1, Some(0.0)).unwrap().evidence()
    );
    assert_eq!(early_second.evidence().source_play_id, "b");
    assert_eq!(late_first.evidence().source_play_id, "b");
    assert_eq!(early_second.materialize_prefix("b0", 1).unwrap(), [6]);
    assert_eq!(late_first.materialize_prefix("b0", 1).unwrap(), [10]);
    assert_ne!(
        late_first.identity("b0").unwrap().cache_id,
        early_second.identity("b0").unwrap().cache_id
    );
    assert_eq!(
        first
            .snapshots()
            .iter()
            .map(|snapshot| snapshot.t_star_ms.to_bits())
            .collect::<Vec<_>>(),
        [
            4631854218153394059,
            4634544125269611050,
            4632905092361953415
        ],
        "versioned seed sampling must retain its exact binary64 cuts"
    );
}

#[test]
fn single_start_and_missing_or_zero_api_duration_preserve_provenance() {
    let single = graph(vec![row("only", "main", 0.0, None)]);
    let prepared = single
        .prepare_snapshots(2, AgenticSnapshotOptions { seed: 3 })
        .unwrap();
    for snapshot in prepared.snapshots() {
        assert_eq!(snapshot.t_star_ms, 0.0);
        assert!(!snapshot.requests[0].historical);
        assert_eq!(snapshot.requests[0].recorded_end_ms, None);
        assert!(snapshot.primers.is_empty());
    }
    for duration in [None, Some(0.0)] {
        let predecessor = row("before", "main", 0.0, duration);
        let mut after = row("after", "main", 100.0, None);
        after.dependencies.push(edge(
            "before",
            AgenticDependencyRelation::Sequence,
            AgenticDependencyTrigger::Completion,
            150.0,
        ));
        let graph = graph(vec![predecessor, after]);
        let play = prepare(&graph, 50.0);
        assert_eq!(request(&play, "before").recorded_end_ms, duration);
        assert_eq!(request(&play, "after").remaining_delay_ms, 100.0);
    }
}

#[test]
fn snapshots_retain_full_graph_outputs_and_logical_prompt_views() {
    let graph = boundary_graph();
    let prepared = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 8 })
        .unwrap();
    let zero = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
    let cut = prepared.context().prepare_play(0, 0, Some(100.0)).unwrap();
    let full = drain(WorkloadDriver::new_agentic_trace(graph.clone(), 64).unwrap());
    let at_zero = drain(
        WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![zero]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap(),
    );
    let suffix = drain(
        WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![cut.clone()]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap(),
    );
    assert_eq!(at_zero.len(), full.len());
    for (id, ready) in &full {
        assert_eq!(ready.request.tokens, at_zero[id].request.tokens);
        assert_eq!(ready.replay_hashes, at_zero[id].replay_hashes);
        assert_eq!(
            ready.request.output_token_ids,
            at_zero[id].request.output_token_ids
        );
    }
    for (id, ready) in &suffix {
        assert_eq!(
            ready.request.output_token_ids,
            full[id].request.output_token_ids
        );
        assert_eq!(ready.request_uuid, at_zero[id].request_uuid);
        assert_eq!(
            ready.request.tokens,
            cut.materialize_prefix(id, 128).unwrap()
        );
    }
}

#[test]
fn original_weka_prefix_keeps_shared_units_and_private_partial_tail() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.json");
    let requests = [(223, vec![10, 20, 30, 40]), (80, vec![10, 20]), (223, vec![10, 20, 30, 40])].into_iter().enumerate().map(|(index, (length, hashes))| {
        serde_json::json!({"t": index, "type":"s", "model":"model", "in":length, "out":1, "hash_ids":hashes})
    }).collect::<Vec<_>>();
    std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"id":"play", "models":["model"], "block_size":64, "hash_id_scope":"local", "requests":requests})).unwrap()).unwrap();
    let graph = load_weka_agentic_graph(&path, Some(64)).unwrap();
    let id = |outer| {
        graph
            .nodes()
            .iter()
            .find(|node| node.request_id().ends_with(&format!(":outer:{outer}")))
            .unwrap()
            .request_id()
            .to_owned()
    };
    let original = id(0);
    let shortened = id(1);
    let follower = id(2);
    let play = prepare(&graph, 1000.0);
    let full = play.materialize_prefix(&original, 223).unwrap();
    let independently_short = play.materialize_prefix(&shortened, 80).unwrap();
    assert_eq!(&full[..64], &independently_short[..64]);
    assert_ne!(&full[64..80], &independently_short[64..]);
    let following = play.materialize_prefix(&follower, 223).unwrap();
    assert_eq!(&full[..192], &following[..192]);
    assert_ne!(&full[192..], &following[192..]);
    for length in [0, 1, 63, 64, 65, 80, 127, 128, 129, 192, 193, 207, 222, 223] {
        assert_eq!(
            play.materialize_prefix(&original, length).unwrap(),
            full[..length]
        );
        for block_size in [16, 32, 48, 64, 96, 128, 256] {
            assert_eq!(
                play.replay_hashes(&original, length, block_size).unwrap(),
                ReplayRequestHashes::from_tokens(&full[..length], block_size as u32)
            );
        }
    }
}

#[test]
fn encounter_allocated_prompt_and_cross_source_engine_blocks_have_literal_goldens() {
    let mut authored = row("golden", "main", 0.0, None);
    authored.input_length = Some(257);
    authored.hash_ids = Some(vec![u64::MAX - 1, 0, 1_u64 << 32, u64::MAX, 7]);
    let graph = graph(vec![authored]);
    let prepared = graph
        .prepare_snapshots(2, AgenticSnapshotOptions { seed: 42 })
        .unwrap();
    let play = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
    let mut expected = [vec![0; 64], vec![1; 64], vec![2; 64], vec![3; 64]].concat();
    expected.push(4);
    assert_eq!(play.materialize_prefix("golden", 257).unwrap(), expected);
    let next = prepared.context().prepare_play(1, 0, Some(0.0)).unwrap();
    assert_eq!(
        next.materialize_prefix("golden", 257).unwrap(),
        expected.iter().map(|token| token + 5).collect::<Vec<_>>()
    );
    // Independently calculated XXH3 seed-1337 literals over little-endian
    // u32 tokens and (parent, local) chains; the 48-token blocks cross 64-token source units.
    let goldens: &[(usize, &[u64], &[u64])] = &[
        (
            48,
            &[
                5668416277218608172,
                14058210135089406444,
                6754173385707188877,
                9744416516818080641,
                10861956436164017793,
            ],
            &[
                5668416277218608172,
                3476548722019456492,
                15586773222372520177,
                12324737480133932217,
                17869052026612087845,
            ],
        ),
        (
            64,
            &[
                15480293642169978529,
                16951273711404654616,
                2439962053643786207,
                15232596819325015918,
            ],
            &[
                15480293642169978529,
                16972514322484578542,
                7711429637831522643,
                15046190559102504818,
            ],
        ),
    ];
    for &(block_size, local, sequence) in goldens {
        let hashes = play.replay_hashes("golden", 257, block_size).unwrap();
        assert_eq!(hashes.local_block_hashes, local);
        assert_eq!(hashes.sequence_hashes, sequence);
    }
}

#[test]
fn invalid_snapshot_inputs_and_exhausted_identity_ranges_fail_before_execution() {
    let graph = cycle_graph();
    assert!(
        graph
            .prepare_snapshots(0, AgenticSnapshotOptions { seed: 1 })
            .is_err()
    );
    let prepared = graph
        .prepare_snapshots(2, AgenticSnapshotOptions { seed: 1 })
        .unwrap();
    let context = prepared.context();
    for cut in [-1.0, 101.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(context.prepare_play(0, 0, Some(cut)).is_err());
    }
    assert!(context.prepare_play(2, 0, None).is_err());
    assert!(context.prepare_play(0, u64::MAX, None).is_err());
    let last = context
        .prepare_play(1, (1_u64 << 30) - 1, Some(0.0))
        .unwrap();
    assert_eq!(last.materialize_prefix("b1", 1).unwrap(), [u32::MAX]);
    assert!(context.prepare_play(0, 1_u64 << 30, Some(0.0)).is_err());
    let play = context.prepare_play(0, 0, Some(0.0)).unwrap();
    assert!(play.materialize_prefix("missing", 0).is_err());
    assert!(play.materialize_prefix("b0", 1).is_err());
    assert!(play.materialize_prefix("a0", 65).is_err());
    assert!(play.replay_hashes("a0", 64, 0).is_err());
    #[cfg(target_pointer_width = "64")]
    assert!(play.replay_hashes("a0", 64, u32::MAX as usize + 1).is_err());
    assert!(PreparedAgenticSnapshots::from_plays(Vec::new()).is_err());
    assert!(PreparedAgenticSnapshots::from_plays(vec![play.clone(), play.clone()]).is_err());
    let independent = graph
        .prepare_snapshots(2, AgenticSnapshotOptions { seed: 1 })
        .unwrap();
    let other_context = independent.context().prepare_play(1, 0, Some(0.0)).unwrap();
    assert!(PreparedAgenticSnapshots::from_plays(vec![play.clone(), other_context]).is_err());
    for speedup in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        let cohort = PreparedAgenticSnapshots::from_plays(vec![play.clone()]).unwrap();
        assert!(WorkloadDriver::new_agentic_snapshots(cohort, 64, true, speedup).is_err());
    }
}

#[test]
fn overflowing_recorded_intervals_and_dependency_deadlines_are_rejected() {
    let interval = graph(vec![
        row("first", "main", 1.0e308, Some(1.0e308)),
        row("last", "main", 1.5e308, None),
    ]);
    assert!(
        interval
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 1 })
            .is_err()
    );
    let root = row("first", "main", 0.0, Some(1.0e308));
    let mut last = row("last", "main", 10.0, None);
    last.dependencies.push(edge(
        "first",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        1.0e308,
    ));
    let deadline = graph(vec![root, last]);
    assert!(
        deadline
            .prepare_snapshots(1, AgenticSnapshotOptions { seed: 1 })
            .is_err()
    );
}

#[test]
fn historical_request_cannot_depend_on_a_request_not_started_at_the_cut() {
    let source = row("later-source", "source", 100.0, Some(1.0));
    let mut historical = row("earlier-target", "target", 0.0, None);
    historical.dependencies.push(edge(
        "later-source",
        AgenticDependencyRelation::Spawn,
        AgenticDependencyTrigger::Dispatch,
        0.0,
    ));
    let graph = graph(vec![source, historical]);
    let error = graph
        .prepare_snapshots(1, AgenticSnapshotOptions { seed: 1 })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("historical request's unresolved dependency")
    );
}
