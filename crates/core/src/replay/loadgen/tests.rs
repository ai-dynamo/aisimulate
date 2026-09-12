// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use tempfile::NamedTempFile;
use uuid::Uuid;

use super::trace::{
    MAX_DECLARED_SEQUENCE_TOKENS, synthesize_trace_tokens, validate_synthesizable_prompt,
};
use super::*;
use rand::SeedableRng;

/// First eight synthetic output token IDs from a freshly seeded plan RNG.
const GOLDEN_PLAN: [u32; 8] = [
    952584220, 694593853, 2967053780, 2932165014, 2184908278, 3971079078, 3124715654, 3718916919,
];

fn write_trace(lines: &[serde_json::Value]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for line in lines {
        use std::io::Write;
        writeln!(file, "{}", serde_json::to_string(line).unwrap()).unwrap();
    }
    file
}

fn write_agentic_trace(rows: &[serde_json::Value]) -> NamedTempFile {
    let mut lines = vec![serde_json::json!({
        "schema": AGENTIC_MOONCAKE_SCHEMA,
        "version": AGENTIC_MOONCAKE_VERSION,
        "block_size": 4,
        "hash_id_scope": "local",
        "source": {"format": "test", "digest": "fixture"}
    })];
    lines.extend_from_slice(rows);
    write_trace(&lines)
}

#[test]
fn trace_synthesis_bounds_each_hash_block_to_remaining_input() {
    let tokens = synthesize_trace_tokens(1, &[7], usize::MAX).unwrap();
    assert_eq!(tokens, vec![7]);
}

#[test]
fn trace_synthesis_rejects_capacity_overflow() {
    let error = validate_synthesizable_prompt(1, &[7, 8], usize::MAX).unwrap_err();
    assert!(error.to_string().contains("capacity overflow"));
}

#[test]
fn trace_file_cardinality_validation_is_format_neutral() {
    // The *empty* rejection is format-neutral, so assert it on both a
    // single-file format and the multi-file one.
    for format in [TraceFileFormat::Mooncake, TraceFileFormat::Dynamo] {
        let empty = validate_trace_files(format, &[]).unwrap_err();
        assert!(
            empty.to_string().contains("at least one trace file"),
            "{format:?}: unexpected error: {empty}"
        );
    }

    // The *cardinality* rejection is not: Dynamo loads multiple files by
    // design (`from_request_trace_files` merges them), so asserting only the
    // Mooncake arm left the `format != Dynamo` guard free to be deleted.
    let paths = vec!["first.jsonl".into(), "second.jsonl".into()];
    let multiple = validate_trace_files(TraceFileFormat::Mooncake, &paths).unwrap_err();
    assert!(
        multiple
            .to_string()
            .contains("requires exactly one trace file")
    );
    validate_trace_files(TraceFileFormat::Dynamo, &paths)
        .expect("Dynamo accepts multiple trace files");
}

#[test]
fn test_from_mooncake_single_turn_loads_fields_and_canonicalizes_hashes() {
    let file = write_trace(&[serde_json::json!({
        "timestamp": 123.0,
        "input_length": 8,
        "output_length": 4,
        "hash_ids": [7, 8],
        "priority": -3,
        "strict_priority": 7,
    })]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 1);
    let session = &trace.sessions[0];
    assert_eq!(session.first_arrival_timestamp_ms, Some(123.0));
    assert_eq!(session.turns.len(), 1);
    assert_eq!(session.turns[0].input_length, 8);
    assert_eq!(session.turns[0].max_output_tokens, 4);
    assert_eq!(session.turns[0].hash_ids, vec![0, 1]);
    assert_eq!(session.turns[0].priority, -3);
    assert_eq!(session.turns[0].strict_priority, 7);
}

#[test]
fn mooncake_hash_ids_preserve_identity_above_u32() {
    let shared = 0xB300_0000_0000_0000_u64;
    let distinct = 0xB300_0001_0000_0000_u64;
    let file = write_trace(&[
        serde_json::json!({
            "input_length": 1,
            "output_length": 1,
            "hash_ids": [shared],
        }),
        serde_json::json!({
            "input_length": 1,
            "output_length": 1,
            "hash_ids": [distinct],
        }),
        serde_json::json!({
            "input_length": 1,
            "output_length": 1,
            "hash_ids": [shared],
        }),
    ]);

    let trace = Trace::from_mooncake(file.path(), 1).unwrap();
    let requests = trace.to_single_turn_requests().unwrap();

    assert_ne!(requests[0].tokens, requests[1].tokens);
    assert_eq!(requests[0].tokens, requests[2].tokens);
}

#[test]
fn single_turn_requests_plan_missing_output_tokens_deterministically() {
    let trace = Trace {
        block_size: 1,
        sessions: vec![
            SessionTrace {
                session_id: "missing".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 2,
                    max_output_tokens: 3,
                    hash_ids: vec![10, 11],
                    ..Default::default()
                }],
            },
            SessionTrace {
                session_id: "authored".into(),
                first_arrival_timestamp_ms: Some(1.0),
                turns: vec![TurnTrace {
                    input_length: 1,
                    max_output_tokens: 2,
                    output_token_ids: Some(vec![20, 21]),
                    hash_ids: vec![12],
                    ..Default::default()
                }],
            },
        ],
    };

    let first = trace.to_single_turn_requests().unwrap();
    let second = trace.to_single_turn_requests().unwrap();

    assert_eq!(first[0].output_token_ids, second[0].output_token_ids);
    assert_eq!(first[0].output_token_ids.as_ref().map(Vec::len), Some(3));
    assert_eq!(first[1].output_token_ids.as_deref(), Some(&[20, 21][..]));
}

#[test]
fn planned_output_token_ids_matches_its_golden_vector() {
    // The generated output-token plan is part of a byte-exact-parity product,
    // so it needs a pinned vector rather than a self-consistency check. This
    // catches a changed `SYNTHETIC_OUTPUT_SEED`, a swapped PRNG, and a
    // degenerate generator (every ID zero) -- none of which a length-plus-
    // determinism assertion can see.
    let mut rng = rand::rngs::StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
    let plan = planned_output_token_ids(None, 8, &mut rng);
    assert_eq!(plan, GOLDEN_PLAN);

    // The stream continues rather than restarting, so a second draw from the
    // same RNG must not repeat the first.
    let next = planned_output_token_ids(None, 8, &mut rng);
    assert_ne!(next, GOLDEN_PLAN);

    // An authored plan is passed through untouched and draws nothing.
    let mut authored_rng = rand::rngs::StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
    assert_eq!(
        planned_output_token_ids(Some(vec![1, 2, 3]), 8, &mut authored_rng),
        vec![1, 2, 3]
    );
    assert_eq!(
        planned_output_token_ids(None, 8, &mut authored_rng),
        GOLDEN_PLAN
    );
}

#[test]
fn mooncake_rejects_unbounded_declared_output_length() {
    // `output_length` has no `hash_ids` anchor, and every turn's plan is
    // materialized eagerly at driver construction -- so an unbounded value is
    // an allocator abort, not an error, unless it is refused at load.
    let file = write_trace(&[serde_json::json!({
        "input_length": 4,
        "output_length": 1_000_000_000_000_u64,
        "hash_ids": [1],
    })]);

    let error = Trace::from_mooncake(file.path(), 4).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("maximum declared sequence length"),
        "unexpected error: {error}"
    );
}

#[test]
fn agentic_mooncake_rejects_unbounded_declared_output_length() {
    let file = write_agentic_trace(&[serde_json::json!({
        "request_id": "r1",
        "play_id": "p1",
        "session_id": "s1",
        "model": "m",
        "input_length": 4,
        "output_length": 1_000_000_000_000_u64,
        "hash_ids": [1],
        "not_before_ms": 0.0,
    })]);

    let error = AgenticTrace::from_agentic_mooncake(file.path()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("maximum declared sequence length"),
        "unexpected error: {error}"
    );
}

#[test]
fn unvalidated_trace_driver_rejects_unbounded_declared_output_length() {
    let trace = Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "huge".into(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 1,
                // Over the ceiling but still allocatable, so removing the gate
                // fails this test by succeeding rather than by aborting the
                // whole test binary.
                max_output_tokens: MAX_DECLARED_SEQUENCE_TOKENS + 1,
                hash_ids: vec![10],
                ..Default::default()
            }],
        }],
    };

    let error = WorkloadDriver::new_trace(trace, 1).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("maximum declared sequence length"),
        "unexpected error: {error}"
    );
}

#[test]
fn unvalidated_trace_driver_rejects_negative_first_arrival() {
    // `new_trace` skips `Trace::validate`, so the in-driver time check is the
    // last gate before a negative origin is stamped into
    // `DirectRequest::arrival_timestamp_ms`.
    let trace = Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "negative".into(),
            first_arrival_timestamp_ms: Some(-100.0),
            turns: vec![TurnTrace {
                input_length: 1,
                max_output_tokens: 1,
                hash_ids: vec![10],
                ..Default::default()
            }],
        }],
    };

    let error = WorkloadDriver::new_trace(trace, 1).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("first_arrival_timestamp_ms") && message.contains("non-negative"),
        "unexpected error: {message}"
    );
}

#[test]
fn test_from_mooncake_preserves_output_token_replay_keys() {
    let file = write_trace(&[
        serde_json::json!({
            "request_id": "explicit",
            "session_id": "s",
            "input_length": 4,
            "output_length": 2,
            "output_token_ids": [10, 11],
            "hash_ids": [1],
        }),
        serde_json::json!({
            "session_id": "s",
            "input_length": 4,
            "output_length": 1,
            "output_token_ids": [12],
            "hash_ids": [2],
        }),
        serde_json::json!({
            "input_length": 4,
            "output_length": 1,
            "output_token_ids": [13],
            "hash_ids": [3],
        }),
    ]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(
        trace.sessions[0].turns[0].replay_key.as_deref(),
        Some("explicit")
    );
    assert_eq!(
        trace.sessions[0].turns[0].output_token_ids.as_deref(),
        Some(&[10, 11][..])
    );
    assert_eq!(
        trace.sessions[0].turns[1].replay_key.as_deref(),
        Some("s:1")
    );
    assert_eq!(
        trace.sessions[0].turns[1].output_token_ids.as_deref(),
        Some(&[12][..])
    );
    assert_eq!(
        trace.sessions[1].turns[0].replay_key.as_deref(),
        Some("line:2")
    );

    let request = trace.sessions[0].turns[0]
        .to_direct_request(4, Uuid::from_u128(1), None)
        .unwrap();
    assert_eq!(request.output_token_ids.as_deref(), Some(&[10, 11][..]));
}

#[test]
fn test_from_mooncake_rejects_output_token_length_mismatch() {
    let file = write_trace(&[serde_json::json!({
        "input_length": 4,
        "output_length": 2,
        "output_token_ids": [10],
        "hash_ids": [1],
    })]);

    let err = Trace::from_mooncake(file.path(), 4).unwrap_err();
    assert!(
        err.to_string()
            .contains("output_length 2 does not match output_token_ids length 1"),
        "{err:#}"
    );
}

#[test]
fn test_trace_validate_rejects_programmatic_output_token_length_mismatch() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "s".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 4,
                max_output_tokens: 2,
                output_token_ids: Some(vec![10]),
                hash_ids: vec![1],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }],
        }],
    };

    let err = trace.validate_for_trace_mode().unwrap_err();
    assert!(
        err.to_string()
            .contains("max_output_tokens 2 does not match output_token_ids length 1"),
        "{err:#}"
    );
}

#[test]
fn test_from_mooncake_multi_turn_uses_session_id_and_delay() {
    let file = write_trace(&[
        serde_json::json!({
            "session_id": "a",
            "timestamp": 10.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1],
        }),
        serde_json::json!({
            "session_id": "a",
            "delay": 25.0,
            "input_length": 8,
            "output_length": 2,
            "hash_ids": [1, 2],
        }),
        serde_json::json!({
            "session_id": "b",
            "timestamp": 20.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [3],
        }),
    ]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 2);
    assert_eq!(trace.sessions[0].session_id, "a");
    assert_eq!(trace.sessions[0].turns.len(), 2);
    assert_eq!(trace.sessions[0].turns[1].delay_after_previous_ms, 25.0);
    assert_eq!(trace.sessions[1].session_id, "b");
}

#[test]
fn test_from_mooncake_defaults_missing_input_length_from_hash_capacity() {
    let file = write_trace(&[serde_json::json!({
        "timestamp": 7.0,
        "output_length": 3,
        "hash_ids": [5, 6],
    })]);

    let trace = Trace::from_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.sessions.len(), 1);
    assert_eq!(trace.sessions[0].turns[0].input_length, 8);
}

#[test]
fn compatible_agentic_loader_preserves_legacy_rows_and_independent_plays() {
    let file = write_trace(&[
        serde_json::json!({
            "request_id": "r1",
            "session_id": "session-a",
            "timestamp": 100.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1]
        }),
        serde_json::json!({
            "request_id": "r2",
            "session_id": "session-a",
            "wait_for": ["r1"],
            "delay": 10.0,
            "tool_wait_ms": 6.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1]
        }),
        serde_json::json!({
            "request_id": "r3",
            "session_id": "session-b",
            "timestamp": 120.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [2]
        }),
    ]);

    let trace = load_agentic_mooncake(file.path(), 4).unwrap();

    assert_eq!(trace.node_count(), 3);
    assert_eq!(trace.play_count(), 2);
    assert_eq!(trace.nodes()[1].dependencies()[0].delay_ms, 16.0);
}

/// A negative authored `delay` must be rejected on its own, not netted out
/// against `tool_wait_ms`.
///
/// Only the sum was validated, so `delay: -5.0` with `tool_wait_ms: 10.0` was
/// accepted as 5.0 and every dependency edge from that row carried a delay the
/// trace never authored. A larger negative did fail, but blamed the summed
/// "dependency delay" rather than the field that was wrong.
#[test]
fn compatible_agentic_loader_rejects_a_negative_authored_delay() {
    let row = |delay: f64| {
        write_trace(&[
            serde_json::json!({
                "request_id": "r1", "timestamp": 0.0,
                "input_length": 4, "output_length": 1, "hash_ids": [1]
            }),
            serde_json::json!({
                "request_id": "r2", "wait_for": ["r1"],
                "delay": delay, "tool_wait_ms": 10.0,
                "input_length": 4, "output_length": 1, "hash_ids": [1]
            }),
        ])
    };

    // Previously absorbed into a 5.0 edge.
    let file = row(-5.0);
    let error = load_agentic_mooncake(file.path(), 4)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("delay must be finite and nonnegative"),
        "the error must name the authored field, got: {error}"
    );

    let file = row(2.0);
    let trace = load_agentic_mooncake(file.path(), 4).unwrap();
    assert_eq!(trace.nodes()[1].dependencies()[0].delay_ms, 12.0);
}

#[test]
fn compatible_agentic_loader_lowers_multi_root_join_to_one_typed_play() {
    let file = write_trace(&[
        serde_json::json!({
            "request_id": "r1", "timestamp": 0.0,
            "input_length": 4, "output_length": 1, "hash_ids": [1]
        }),
        serde_json::json!({
            "request_id": "r2", "timestamp": 5.0,
            "input_length": 4, "output_length": 1, "hash_ids": [2]
        }),
        serde_json::json!({
            "request_id": "r3", "wait_for": ["r1", "r2"],
            "input_length": 4, "output_length": 1, "hash_ids": [3]
        }),
    ]);

    let trace = load_agentic_mooncake(file.path(), 4).unwrap();

    assert_eq!(trace.play_count(), 1);
    assert_eq!(trace.node_count(), 3);
    assert!(trace.nodes()[1].dependencies().is_empty());
    assert_eq!(trace.nodes()[2].dependencies().len(), 2);

    let mut driver = WorkloadDriver::new_agentic_trace(trace, 4).unwrap();
    let first = driver.pop_ready(0.0, usize::MAX);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].authored_request_id.as_deref(), Some("r1"));
    let second = driver.pop_ready(5.0, usize::MAX);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].authored_request_id.as_deref(), Some("r2"));
}

#[test]
fn test_from_agentic_mooncake_builds_typed_graph() {
    let file = write_agentic_trace(&[
        serde_json::json!({
            "request_id": "r1",
            "play_id": "play",
            "session_id": "root",
            "model": "model",
            "not_before_ms": 0.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1],
            "priority": 5,
            "strict_priority": 6
        }),
        serde_json::json!({
            "request_id": "r2",
            "play_id": "play",
            "session_id": "root",
            "model": "model",
            "not_before_ms": 100.0,
            "dependencies": [{
                "request_id": "r1",
                "trigger": "dispatch",
                "delay_ms": 12.0,
                "relation": "spawn"
            }],
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1]
        }),
    ]);

    let trace = AgenticTrace::from_agentic_mooncake(file.path()).unwrap();
    assert_eq!(trace.nodes.len(), 2);
    assert_eq!(trace.nodes[0].request_id, "r1");
    assert_eq!(trace.nodes[0].priority, 5);
    assert_eq!(trace.nodes[0].strict_priority, 6);
    assert_eq!(trace.nodes[1].dependencies.len(), 1);
    assert_eq!(
        trace.nodes[1].dependencies[0].trigger,
        AgenticDependencyTrigger::Dispatch
    );
    assert_eq!(trace.nodes[1].dependencies[0].delay_ms, 12.0);
    assert_eq!(trace.plays.len(), 1);
    assert_eq!(trace.plays[0].root_nodes, vec![0]);
}

#[test]
fn test_from_agentic_mooncake_rejects_unknown_dependency() {
    let file = write_agentic_trace(&[serde_json::json!({
        "request_id": "r1",
        "play_id": "play",
        "session_id": "root",
        "model": "model",
        "not_before_ms": 0.0,
        "dependencies": [{
            "request_id": "missing",
            "trigger": "completion",
            "delay_ms": 0.0,
            "relation": "join"
        }],
        "input_length": 4,
        "output_length": 1,
        "hash_ids": [1]
    })]);

    let err = AgenticTrace::from_agentic_mooncake(file.path()).unwrap_err();
    assert!(err.to_string().contains("unknown request_id"));
}

#[test]
fn test_from_agentic_mooncake_rejects_input_length_above_hash_capacity() {
    let file = write_agentic_trace(&[serde_json::json!({
        "request_id": "r1",
        "play_id": "play",
        "session_id": "root",
        "model": "model",
        "not_before_ms": 0.0,
        "input_length": 9,
        "output_length": 1,
        "hash_ids": [1, 2]
    })]);

    let err = AgenticTrace::from_agentic_mooncake(file.path()).unwrap_err();
    assert!(err.to_string().contains("input_length 9"));
}

#[test]
fn agentic_v2_rejects_invalid_schema_and_graph_contracts() {
    enum Fixture {
        Raw(Vec<serde_json::Value>),
        Rows(Vec<serde_json::Value>),
    }

    let node = |request_id: &str, play_id: &str, dependencies: serde_json::Value| {
        serde_json::json!({
            "request_id": request_id,
            "play_id": play_id,
            "session_id": "session",
            "model": "model",
            "not_before_ms": 0.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1],
            "dependencies": dependencies
        })
    };
    let dependency = |request_id: &str| {
        serde_json::json!({
            "request_id": request_id,
            "trigger": "completion",
            "delay_ms": 0.0,
            "relation": "sequence"
        })
    };
    let cases = [
        (
            "headerless",
            Fixture::Raw(vec![node("r1", "p", serde_json::json!([]))]),
            "v2 header",
        ),
        (
            "unknown version",
            Fixture::Raw(vec![serde_json::json!({
                "schema": AGENTIC_MOONCAKE_SCHEMA,
                "version": AGENTIC_MOONCAKE_VERSION + 1,
                "block_size": 4,
                "hash_id_scope": "local",
                "source": {"format": "test", "digest": "fixture"}
            })]),
            "unsupported agentic Mooncake version",
        ),
        (
            "duplicate request",
            Fixture::Rows(vec![
                node("r1", "p", serde_json::json!([])),
                node("r1", "p", serde_json::json!([])),
            ]),
            "duplicates request_id",
        ),
        (
            "cycle",
            Fixture::Rows(vec![
                node("r1", "p", serde_json::json!([dependency("r2")])),
                node("r2", "p", serde_json::json!([dependency("r1")])),
            ]),
            "cycle detected",
        ),
        (
            "cross-play dependency",
            Fixture::Rows(vec![
                node("r1", "p1", serde_json::json!([])),
                node("r2", "p2", serde_json::json!([dependency("r1")])),
            ]),
            "depends on request r1 in play p1",
        ),
        (
            "invalid timing",
            Fixture::Rows(vec![{
                let mut value = node("r1", "p", serde_json::json!([]));
                value["not_before_ms"] = serde_json::json!(-1.0);
                value
            }]),
            "invalid not_before_ms",
        ),
        (
            "invalid typed edge",
            Fixture::Rows(vec![
                node("r1", "p", serde_json::json!([])),
                node(
                    "r2",
                    "p",
                    serde_json::json!([{
                        "request_id": "r1",
                        "trigger": "dispatch",
                        "delay_ms": 0.0,
                        "relation": "join"
                    }]),
                ),
            ]),
            "invalid Join dependency with Dispatch trigger",
        ),
    ];

    for (name, fixture, expected) in cases {
        let file = match fixture {
            Fixture::Raw(lines) => write_trace(&lines),
            Fixture::Rows(rows) => write_agentic_trace(&rows),
        };
        let error = AgenticTrace::from_agentic_mooncake(file.path()).expect_err(name);
        assert!(
            format!("{error:#}").contains(expected),
            "{name}: unexpected error: {error:#}"
        );
    }
}

#[test]
fn agentic_v2_rejects_ambiguous_source_play_order() {
    let row = |request_id: &str, play_id: &str, ordinal: Option<usize>| {
        let mut row = serde_json::json!({
            "request_id": request_id,
            "play_id": play_id,
            "session_id": play_id,
            "model": "model",
            "not_before_ms": 0.0,
            "input_length": 4,
            "output_length": 1,
            "hash_ids": [1]
        });
        if let Some(ordinal) = ordinal {
            row["source_play_ordinal"] = ordinal.into();
        }
        row
    };
    let cases = [
        (
            "inconsistent rows",
            vec![row("r1", "p1", Some(0)), row("r2", "p1", Some(1))],
            "inconsistent source_play_ordinal",
        ),
        (
            "partially ordered plays",
            vec![row("r1", "p1", Some(0)), row("r2", "p2", None)],
            "must be set for every play",
        ),
        (
            "duplicate ordinals",
            vec![row("r1", "p1", Some(0)), row("r2", "p2", Some(0))],
            "unique and contiguous",
        ),
        (
            "ordinal gap",
            vec![row("r1", "p1", Some(0)), row("r2", "p2", Some(2))],
            "unique and contiguous",
        ),
    ];

    for (name, rows, expected) in cases {
        let file = write_agentic_trace(&rows);
        let error = AgenticTrace::from_agentic_mooncake(file.path()).expect_err(name);
        assert!(
            error.to_string().contains(expected),
            "{name}: unexpected error: {error:#}"
        );
    }
}

#[test]
fn test_from_applied_compute_agentic_expands_rows_into_num_turns_plus_final_request() {
    let file = write_trace(&[serde_json::json!({
        "num_turns": 2,
        "input_prompt_length": 100,
        "assistant_response_length": [10, 20],
        "tool_call_output_length": [30, 40],
        "tool_call_latency": [0.5, 1.25],
        "final_assistant_response_length": 50,
    })]);

    let trace = Trace::from_applied_compute_agentic(file.path(), 64, 0.0, 0).unwrap();
    assert_eq!(trace.sessions.len(), 1);
    let session = &trace.sessions[0];
    assert_eq!(session.first_arrival_timestamp_ms, None);
    assert_eq!(session.turns.len(), 3);
    assert_eq!(session.turns[0].input_length, 100);
    assert_eq!(session.turns[0].max_output_tokens, 10);
    assert_eq!(session.turns[0].delay_after_previous_ms, 0.0);
    assert_eq!(session.turns[1].input_length, 140);
    assert_eq!(session.turns[1].max_output_tokens, 20);
    assert_eq!(session.turns[1].delay_after_previous_ms, 500.0);
    assert_eq!(session.turns[2].input_length, 200);
    assert_eq!(session.turns[2].max_output_tokens, 50);
    assert_eq!(session.turns[2].delay_after_previous_ms, 1250.0);
}

#[test]
fn applied_compute_agentic_rejects_unbounded_declared_input_length() {
    // No `hash_ids` array anchors this loader's declared input length -- it
    // synthesizes one hash per block in an unbounded loop.
    let file = write_trace(&[serde_json::json!({
        "num_turns": 0,
        // Over the ceiling but still importable, so removing the gate fails
        // this test by succeeding rather than by hanging the suite.
        "input_prompt_length": MAX_DECLARED_SEQUENCE_TOKENS + 1,
        "assistant_response_length": [],
        "tool_call_output_length": [],
        "tool_call_latency": [],
        "final_assistant_response_length": 1,
    })]);

    let error = Trace::from_applied_compute_agentic(file.path(), 64, 0.0, 0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("maximum declared sequence length"),
        "unexpected error: {error}"
    );
}

#[test]
fn applied_compute_agentic_rejects_unbounded_cumulative_input_length() {
    // Each turn's response and tool-call output grow the cumulative length,
    // which drives the same synthesis loop on the next turn.
    let file = write_trace(&[serde_json::json!({
        "num_turns": 1,
        "input_prompt_length": 1,
        "assistant_response_length": [MAX_DECLARED_SEQUENCE_TOKENS],
        "tool_call_output_length": [MAX_DECLARED_SEQUENCE_TOKENS],
        "tool_call_latency": [0.0],
        "final_assistant_response_length": 1,
    })]);

    let error = Trace::from_applied_compute_agentic(file.path(), 64, 0.0, 0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cumulative input length 20000001 exceeds"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_from_applied_compute_agentic_prefix_extends_hashes_across_turns() {
    let file = write_trace(&[serde_json::json!({
        "num_turns": 2,
        "input_prompt_length": 600,
        "assistant_response_length": [40, 50],
        "tool_call_output_length": [40, 50],
        "tool_call_latency": [0.1, 0.2],
        "final_assistant_response_length": 60,
    })]);

    let trace = Trace::from_applied_compute_agentic(file.path(), 256, 0.0, 0).unwrap();
    let turns = &trace.sessions[0].turns;
    assert_eq!(turns[0].hash_ids, vec![0, 1, 2]);
    assert_eq!(turns[1].hash_ids, vec![0, 1, 2]);
    assert_eq!(turns[2].hash_ids, vec![0, 1, 2, 3]);
}

#[test]
fn test_from_applied_compute_agentic_can_share_initial_prefix_blocks_across_sessions() {
    let file = write_trace(&[
        serde_json::json!({
            "num_turns": 1,
            "input_prompt_length": 600,
            "assistant_response_length": [10],
            "tool_call_output_length": [10],
            "tool_call_latency": [0.1],
            "final_assistant_response_length": 10,
        }),
        serde_json::json!({
            "num_turns": 1,
            "input_prompt_length": 600,
            "assistant_response_length": [20],
            "tool_call_output_length": [20],
            "tool_call_latency": [0.2],
            "final_assistant_response_length": 20,
        }),
    ]);

    let trace = Trace::from_applied_compute_agentic(file.path(), 256, 0.5, 1).unwrap();
    assert_eq!(
        trace.sessions[0].turns[0].hash_ids[0],
        trace.sessions[1].turns[0].hash_ids[0]
    );
    assert_eq!(
        trace.sessions[0].turns[0].hash_ids[1],
        trace.sessions[1].turns[0].hash_ids[1]
    );
    assert_ne!(
        trace.sessions[0].turns[0].hash_ids[2],
        trace.sessions[1].turns[0].hash_ids[2]
    );
}

#[test]
fn test_turn_to_direct_request_repeats_hash_ids_by_block_size() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        output_token_ids: None,
        replay_key: None,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
        priority: -2,
        strict_priority: 8,
        policy_class: None,
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(1), Some(5.0))
        .unwrap();
    assert_eq!(request.tokens, vec![1, 1, 1, 1, 2, 2]);
    assert_eq!(request.arrival_timestamp_ms, Some(5.0));
    assert_eq!(request.priority, -2);
    assert_eq!(request.strict_priority, 8);
}

#[test]
fn test_turn_replay_hashes_match_full_blocks_only() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
        ..Default::default()
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(1), Some(5.0))
        .unwrap();
    let replay_hashes = turn.to_replay_hashes(4, 4).unwrap();
    assert_eq!(
        replay_hashes,
        ReplayRequestHashes::from_tokens(&request.tokens, 4)
    );
    assert_eq!(replay_hashes.local_block_hashes.len(), 1);
}

#[test]
fn test_turn_replay_hashes_support_distinct_trace_and_engine_block_sizes() {
    let turn = TurnTrace {
        input_length: 6,
        max_output_tokens: 3,
        hash_ids: vec![1, 2],
        delay_after_previous_ms: 0.0,
        ..Default::default()
    };

    let request = turn
        .to_direct_request(4, Uuid::from_u128(2), Some(5.0))
        .unwrap();
    let replay_hashes = turn.to_replay_hashes(4, 2).unwrap();
    assert_eq!(
        replay_hashes,
        ReplayRequestHashes::from_tokens(&request.tokens, 2)
    );
    assert_eq!(replay_hashes.local_block_hashes.len(), 3);
}

#[test]
fn test_partition_by_session_round_robin_keeps_sessions_intact() {
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 4,
        turns_per_session: 2,
        input_tokens: LengthSpec {
            mean: 8,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.5,
        num_prefix_groups: 2,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::ConstantMs(5.0),
        seed: 7,
        arrival_seed: 42,
    })
    .unwrap();

    let partitions = trace
        .partition_by_session(SessionPartitionSpec::RoundRobin { num_partitions: 2 })
        .unwrap();
    assert_eq!(partitions.len(), 2);
    // Assert session *identity*, not just cardinality: a block partitioner
    // (`session_idx / num_partitions`) also yields two partitions of two, so
    // counts alone cannot tell round-robin from block striping.
    let session_ids = |partition: &Trace| {
        partition
            .sessions
            .iter()
            .map(|session| session.session_id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(session_ids(&partitions[0]), vec!["session_0", "session_2"]);
    assert_eq!(session_ids(&partitions[1]), vec!["session_1", "session_3"]);
    assert!(
        partitions
            .iter()
            .flat_map(|partition| partition.sessions.iter())
            .all(|session| session.turns.len() == 2)
    );
}

fn single_session_trace(first_arrival_timestamp_ms: Option<f64>) -> Trace {
    Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "only".into(),
            first_arrival_timestamp_ms,
            turns: vec![TurnTrace {
                input_length: 1,
                max_output_tokens: 1,
                hash_ids: vec![1],
                ..Default::default()
            }],
        }],
    }
}

#[test]
fn speed_up_timing_refuses_a_ratio_that_overflows_a_timestamp() {
    // The ratio is finite and positive, so the existing argument check admits
    // it -- but a denormal divisor sends a finite timestamp to +inf.
    let error = single_session_trace(Some(1.0e300))
        .speed_up_timing(f64::MIN_POSITIVE)
        .unwrap_err();
    assert!(
        error.to_string().contains("is not finite after speeding up"),
        "unexpected error: {error}"
    );
}

#[test]
fn rescale_session_start_span_refuses_a_non_finite_span() {
    // `min == max == +inf` makes the span NaN, which is neither `== 0.0` nor a
    // usable divisor: every rescaled timestamp silently became NaN.
    let mut trace = single_session_trace(Some(f64::INFINITY));
    trace.sessions.push(SessionTrace {
        session_id: "second".into(),
        first_arrival_timestamp_ms: Some(f64::INFINITY),
        turns: trace.sessions[0].turns.clone(),
    });

    let error = trace.rescale_session_start_span(1_000).unwrap_err();
    assert!(
        error.to_string().contains("span is not finite"),
        "unexpected error: {error}"
    );
}

#[test]
fn partition_by_session_refuses_a_zero_partition_count() {
    // `.max(1)` used to repair this silently, handing back one partition
    // containing the whole trace as if the split had succeeded.
    let trace = Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "only".into(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 1,
                max_output_tokens: 1,
                hash_ids: vec![1],
                ..Default::default()
            }],
        }],
    };

    let error = trace
        .partition_by_session(SessionPartitionSpec::RoundRobin { num_partitions: 0 })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("num_partitions must be greater than 0"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_synthetic_prefix_groups_share_prefixes_within_group() {
    let trace = Trace::synthetic(SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 6,
        turns_per_session: 1,
        input_tokens: LengthSpec {
            mean: 16,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.5,
        num_prefix_groups: 2,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::None,
        seed: 42,
        arrival_seed: 42,
    })
    .unwrap();

    let prefixes = trace
        .sessions
        .iter()
        .map(|session| session.turns[0].hash_ids[..2].to_vec())
        .collect::<HashSet<_>>();
    let suffixes = trace
        .sessions
        .iter()
        .map(|session| session.turns[0].hash_ids[2..].to_vec())
        .collect::<HashSet<_>>();

    assert_eq!(prefixes.len(), 2);
    assert_eq!(suffixes.len(), trace.sessions.len());
}

/// `shared_prefix_ratio` and `num_prefix_groups` are authored independently and
/// default independently, so asking for shared prefixes while leaving the group
/// count at zero is an easy mistake. It used to be an invisible one: the
/// prefix-block loop pushed nothing, the backfill gave every block a unique
/// hash, and the run silently measured a 0%-shared workload.
/// A NaN stddev used to make every Box-Muller draw NaN, and `f64::max` returns
/// the non-NaN operand -- so the authored workload was silently replaced by
/// single-token turns. A huge stddev saturated the `as usize` cast instead of
/// wrapping, turning into a `Vec::with_capacity` abort.
#[test]
fn test_synthetic_rejects_unusable_length_distributions() {
    let spec = |input_tokens: LengthSpec| SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 2,
        turns_per_session: 1,
        input_tokens,
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.0,
        num_prefix_groups: 0,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::None,
        seed: 42,
        arrival_seed: 42,
    };

    for stddev in [f64::NAN, f64::INFINITY] {
        let error = Trace::synthetic(spec(LengthSpec { mean: 16, stddev }))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("input_tokens"),
            "the error must name the offending field, got: {error}"
        );
    }

    let error = Trace::synthetic(spec(LengthSpec {
        mean: usize::MAX,
        stddev: 0.0,
    }))
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("input_tokens mean"),
        "an unbounded mean must be refused, got: {error}"
    );

    // A finite, ordinary distribution must stay accepted.
    Trace::synthetic(spec(LengthSpec {
        mean: 16,
        stddev: 4.0,
    }))
    .expect("a finite length distribution must remain accepted");
}

#[test]
fn test_synthetic_shared_prefix_ratio_without_groups_is_rejected() {
    let spec = |num_prefix_groups| SyntheticTraceSpec {
        block_size: 4,
        num_sessions: 2,
        turns_per_session: 1,
        input_tokens: LengthSpec {
            mean: 16,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: 2,
            stddev: 0.0,
        },
        shared_prefix_ratio: 0.5,
        num_prefix_groups,
        first_turn_arrivals: ArrivalSpec::Burst,
        inter_turn_delays: DelaySpec::None,
        seed: 42,
        arrival_seed: 42,
    };

    let error = Trace::synthetic(spec(0)).unwrap_err().to_string();
    assert!(
        error.contains("num_prefix_groups"),
        "the error must name the missing knob, got: {error}"
    );

    // A zero ratio genuinely does not need groups, and must stay accepted.
    Trace::synthetic(SyntheticTraceSpec {
        shared_prefix_ratio: 0.0,
        ..spec(0)
    })
    .unwrap();
}

#[test]
fn test_synthetic_arrival_mode_changes_timestamps_only() {
    let build = |first_turn_arrivals, arrival_seed| {
        Trace::synthetic(SyntheticTraceSpec {
            block_size: 4,
            num_sessions: 20,
            turns_per_session: 3,
            input_tokens: LengthSpec {
                mean: 16,
                stddev: 3.0,
            },
            output_tokens: LengthSpec {
                mean: 4,
                stddev: 1.0,
            },
            shared_prefix_ratio: 0.5,
            num_prefix_groups: 5,
            first_turn_arrivals,
            inter_turn_delays: DelaySpec::ExponentialMs { mean_ms: 8.0 },
            seed: 99,
            arrival_seed,
        })
        .unwrap()
    };

    let strip_timestamps = |trace: &mut Trace| {
        trace
            .sessions
            .iter_mut()
            .map(|session| session.first_arrival_timestamp_ms.take())
            .collect::<Vec<_>>()
    };

    let mut fixed = build(ArrivalSpec::ConstantQps { qps: 25.0 }, 42);
    let mut poisson = build(ArrivalSpec::PoissonQps { qps: 25.0 }, 42);
    let mut reseeded_poisson = build(ArrivalSpec::PoissonQps { qps: 25.0 }, 7);
    let fixed_timestamps = strip_timestamps(&mut fixed);
    let poisson_timestamps = strip_timestamps(&mut poisson);
    let reseeded_timestamps = strip_timestamps(&mut reseeded_poisson);
    assert_ne!(fixed_timestamps, poisson_timestamps);
    assert_ne!(poisson_timestamps, reseeded_timestamps);
    assert_eq!(fixed, poisson);
    assert_eq!(poisson, reseeded_poisson);
}

#[test]
fn test_expand_hash_prefix_depth_scales_hashes_and_input_length() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "session".to_string(),
            first_arrival_timestamp_ms: Some(10.0),
            turns: vec![TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![7, 8],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }],
        }],
    }
    .expand_hash_prefix_depth(3)
    .unwrap();

    let turn = &trace.sessions[0].turns[0];
    assert_eq!(turn.input_length, 18);
    assert_eq!(turn.hash_ids, vec![21, 22, 23, 24, 25, 26]);

    let request = turn
        .to_direct_request(trace.block_size, Uuid::from_u128(2), Some(10.0))
        .unwrap();
    assert_eq!(request.tokens.len(), 18);
}

#[test]
fn test_expand_hash_prefix_depth_rejects_offset_overflow() {
    let error = Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "session".to_string(),
            first_arrival_timestamp_ms: None,
            turns: vec![TurnTrace {
                input_length: 1,
                max_output_tokens: 1,
                hash_ids: vec![u32::MAX / 3],
                ..Default::default()
            }],
        }],
    }
    .expand_hash_prefix_depth(3)
    .unwrap_err();
    assert!(
        error.to_string().contains("hash prefix expansion overflow"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_rescale_ready_span_scales_session_starts_and_inter_turn_delays() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![
            SessionTrace {
                session_id: "a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 20.0,
                        ..Default::default()
                    },
                ],
            },
            SessionTrace {
                session_id: "b".to_string(),
                first_arrival_timestamp_ms: Some(30.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![3],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            },
        ],
    }
    .rescale_ready_span(100)
    .unwrap();

    assert_eq!(trace.sessions[0].first_arrival_timestamp_ms, Some(0.0));
    assert_eq!(trace.sessions[1].first_arrival_timestamp_ms, Some(100.0));
    assert_eq!(trace.sessions[0].turns[1].delay_after_previous_ms, 100.0);
}

/// A sub-millisecond source span must still rescale to the requested duration.
///
/// The divide-by-zero guard was a floor on the denominator
/// (`(max - min).max(1.0)`), not a special case, so a 0.4ms span divided by 1.0
/// and the trace was rescaled to 0.4 * duration_ms instead of duration_ms --
/// the function's entire contract violated by 2.5x, silently, for every span
/// under 1ms.
#[test]
fn test_rescale_ready_span_honors_a_sub_millisecond_source_span() {
    let session = |session_id: &str, start_ms: f64, hash_id: u32| SessionTrace {
        session_id: session_id.to_string(),
        first_arrival_timestamp_ms: Some(start_ms),
        turns: vec![TurnTrace {
            input_length: 4,
            max_output_tokens: 1,
            hash_ids: vec![hash_id],
            delay_after_previous_ms: 0.0,
            ..Default::default()
        }],
    };

    let trace = Trace {
        block_size: 4,
        sessions: vec![session("a", 0.0, 1), session("b", 0.4, 2)],
    }
    .rescale_ready_span(100)
    .unwrap();

    assert_eq!(trace.sessions[0].first_arrival_timestamp_ms, Some(0.0));
    assert_eq!(
        trace.sessions[1].first_arrival_timestamp_ms,
        Some(100.0),
        "the rescaled span must be the requested duration, not 0.4 of it"
    );

    // A genuinely zero span still collapses rather than dividing by zero,
    // matching the sibling rescale_session_start_span.
    let collapsed = Trace {
        block_size: 4,
        sessions: vec![session("a", 5.0, 1), session("b", 5.0, 2)],
    }
    .rescale_ready_span(100)
    .unwrap();
    assert_eq!(collapsed.sessions[0].first_arrival_timestamp_ms, Some(0.0));
    assert_eq!(collapsed.sessions[1].first_arrival_timestamp_ms, Some(0.0));
}

#[test]
fn test_driver_requires_completion_before_follow_up_turn() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "s".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![2],
                    delay_after_previous_ms: 10.0,
                    ..Default::default()
                },
            ],
        }],
    };

    let mut driver = trace.into_trace_driver().unwrap();
    let first = driver.pop_ready(0.0, 1);
    assert_eq!(first.len(), 1);
    assert!(driver.pop_ready(100.0, 1).is_empty());

    driver.on_complete(first[0].request_uuid, 5.0).unwrap();
    assert!(driver.pop_ready(14.0, 1).is_empty());
    let second = driver.pop_ready(15.0, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].turn_index, 1);
}

#[test]
fn test_driver_next_ready_time_tracks_earliest_pending_turn() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![
            SessionTrace {
                session_id: "a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![1],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 1,
                        hash_ids: vec![2],
                        delay_after_previous_ms: 5.0,
                        ..Default::default()
                    },
                ],
            },
            SessionTrace {
                session_id: "b".to_string(),
                first_arrival_timestamp_ms: Some(20.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![3],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            },
        ],
    };

    let mut driver = trace.into_trace_driver().unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(10.0));

    let first = driver.pop_ready(10.0, 1);
    assert_eq!(first.len(), 1);
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));

    driver.on_complete(first[0].request_uuid, 25.0).unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));

    let second = driver.pop_ready(20.0, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(driver.next_ready_time_ms(), Some(30.0));
}

#[test]
fn test_trace_driver_round_trips_turn_semantics_into_ready_requests() {
    let trace = Trace {
        block_size: 2,
        sessions: vec![
            SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(10.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![1, 2],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 2,
                        max_output_tokens: 3,
                        hash_ids: vec![3],
                        delay_after_previous_ms: 5.0,
                        ..Default::default()
                    },
                ],
            },
            SessionTrace {
                session_id: "session-b".to_string(),
                first_arrival_timestamp_ms: Some(12.0),
                turns: vec![TurnTrace {
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![4],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            },
        ],
    };
    let expected = trace.clone();
    let mut driver = trace.into_trace_driver().unwrap();

    assert!(driver.pop_ready(9.0, usize::MAX).is_empty());

    let first = driver.pop_ready(10.0, usize::MAX);
    assert_eq!(first.len(), 1);
    let first = &first[0];
    assert_eq!(first.session_id, "session-a");
    assert_eq!(first.turn_index, 0);
    assert_eq!(first.scheduled_ready_at_ms, 10.0);
    assert_eq!(
        first.request.tokens.len(),
        expected.sessions[0].turns[0].input_length
    );
    assert_eq!(
        first.request.max_output_tokens,
        expected.sessions[0].turns[0].max_output_tokens
    );
    assert_eq!(first.request.arrival_timestamp_ms, Some(10.0));
    assert_eq!(
        first.replay_hashes.as_ref(),
        Some(
            &expected.sessions[0].turns[0]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );
    let expected_first_request = expected.sessions[0].turns[0]
        .to_direct_request(expected.block_size, first.request_uuid, Some(10.0))
        .unwrap();
    assert_eq!(first.request.tokens, expected_first_request.tokens);
    assert_eq!(
        first.request.max_output_tokens,
        expected_first_request.max_output_tokens
    );
    assert_eq!(first.request.uuid, expected_first_request.uuid);
    assert_eq!(
        first.request.arrival_timestamp_ms,
        expected_first_request.arrival_timestamp_ms
    );

    let second = driver.pop_ready(12.0, usize::MAX);
    assert_eq!(second.len(), 1);
    let second = &second[0];
    assert_eq!(second.session_id, "session-b");
    assert_eq!(second.turn_index, 0);
    assert_eq!(second.scheduled_ready_at_ms, 12.0);
    assert_eq!(
        second.request.tokens.len(),
        expected.sessions[1].turns[0].input_length
    );
    assert_eq!(
        second.request.max_output_tokens,
        expected.sessions[1].turns[0].max_output_tokens
    );
    assert_eq!(second.request.arrival_timestamp_ms, Some(12.0));
    assert_eq!(
        second.replay_hashes.as_ref(),
        Some(
            &expected.sessions[1].turns[0]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );

    driver.on_complete(first.request_uuid, 20.0).unwrap();
    assert!(driver.pop_ready(24.0, usize::MAX).is_empty());

    let third = driver.pop_ready(25.0, usize::MAX);
    assert_eq!(third.len(), 1);
    let third = &third[0];
    assert_eq!(third.session_id, "session-a");
    assert_eq!(third.turn_index, 1);
    assert_eq!(third.scheduled_ready_at_ms, 25.0);
    assert_eq!(
        third.request.tokens.len(),
        expected.sessions[0].turns[1].input_length
    );
    assert_eq!(
        third.request.max_output_tokens,
        expected.sessions[0].turns[1].max_output_tokens
    );
    assert_eq!(third.request.arrival_timestamp_ms, Some(25.0));
    assert_eq!(
        third.replay_hashes.as_ref(),
        Some(
            &expected.sessions[0].turns[1]
                .to_replay_hashes(expected.block_size, expected.block_size)
                .unwrap()
        )
    );
    let expected_third_request = expected.sessions[0].turns[1]
        .to_direct_request(expected.block_size, third.request_uuid, Some(25.0))
        .unwrap();
    assert_eq!(third.request.tokens, expected_third_request.tokens);
    assert_eq!(
        third.request.max_output_tokens,
        expected_third_request.max_output_tokens
    );
    assert_eq!(third.request.uuid, expected_third_request.uuid);
    assert_eq!(
        third.request.arrival_timestamp_ms,
        expected_third_request.arrival_timestamp_ms
    );
}

#[test]
fn test_trace_driver_rechunks_trace_blocks_into_engine_blocks() {
    let trace = Trace {
        block_size: 4,
        sessions: vec![SessionTrace {
            session_id: "session-a".to_string(),
            first_arrival_timestamp_ms: Some(10.0),
            turns: vec![TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![1, 2],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }],
        }],
    };
    let mut driver = trace.into_trace_driver_with_block_size(2).unwrap();

    let ready = driver.pop_ready(10.0, usize::MAX);
    assert_eq!(ready.len(), 1);
    let ready = &ready[0];
    assert_eq!(ready.request.tokens, vec![1, 1, 1, 1, 2, 2]);
    assert_eq!(
        ready.replay_hashes.as_ref(),
        Some(
            &TurnTrace {
                input_length: 6,
                max_output_tokens: 2,
                hash_ids: vec![1, 2],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }
            .to_replay_hashes(4, 2)
            .unwrap()
        )
    );
}
