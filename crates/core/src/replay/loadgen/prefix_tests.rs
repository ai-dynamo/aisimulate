// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-authored source-identity fixtures, through Weka import, driver routing,
//! and the engine's independent block/sequence hashing implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use tempfile::tempdir;

use super::*;
use crate::engine::{compute_block_hash_for_seq, compute_seq_hash_for_block};

const SOURCE_BLOCK_SIZE: usize = 64;
const ENGINE_BLOCK_SIZES: [usize; 7] = [16, 32, 48, 64, 96, 128, 256];

fn weka_request(input_length: usize, hash_ids: &[u64]) -> serde_json::Value {
    serde_json::json!({
        "t": 0.0, "type": "s", "model": "model",
        "in": input_length, "out": 1, "hash_ids": hash_ids
    })
}

fn weka_requests(id: &str, requests: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "id": id, "models": ["model"], "block_size": SOURCE_BLOCK_SIZE,
        "hash_id_scope": "local", "requests": requests
    })
}

fn weka_play(id: &str, input_length: usize, hash_ids: &[u64]) -> serde_json::Value {
    weka_requests(id, vec![weka_request(input_length, hash_ids)])
}

fn write_weka(path: &Path, plays: &[serde_json::Value]) {
    let text = plays
        .iter()
        .map(|play| serde_json::to_string(play).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, text + "\n").unwrap();
}

fn weka_node<'a>(graph: &'a ValidatedAgenticGraph, play: &str, outer: usize) -> &'a AgenticNode {
    graph
        .nodes()
        .iter()
        .find(|node| {
            node.play_id().ends_with(&format!(":play:{play}"))
                && node
                    .request_id()
                    .ends_with(&format!(":request:outer:{outer}"))
        })
        .unwrap()
}

fn engine_hashes(tokens: &[u32], block_size: usize) -> ReplayRequestHashes {
    let local = compute_block_hash_for_seq(tokens, block_size);
    ReplayRequestHashes {
        sequence_hashes: compute_seq_hash_for_block(&local),
        local_block_hashes: local.into_iter().map(|value| value.0).collect(),
    }
}

fn routed_requests(
    graph: &ValidatedAgenticGraph,
    block_size: usize,
) -> BTreeMap<String, ReadyTurn> {
    let mut driver = WorkloadDriver::new_agentic_trace(graph.clone(), block_size).unwrap();
    let mut requests = BTreeMap::new();
    // Drive only the authored dependency graph here; actual engine admission
    // and cache reuse are exercised by the aggregated replay suite.
    while !driver.is_drained() {
        let now = driver
            .next_ready_time_ms()
            .expect("fixture graph must make progress");
        let ready = driver.pop_ready(now, usize::MAX);
        assert!(!ready.is_empty());
        for ready in ready {
            driver.on_complete(ready.request_uuid, now).unwrap();
            requests.insert(ready.authored_request_id.clone().unwrap(), ready);
        }
    }
    assert_eq!(requests.len(), graph.node_count());
    requests
}

#[test]
fn weka_reblocking_boundaries_match_materialized_graph_and_engine_hashes() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.jsonl");
    let lengths = ENGINE_BLOCK_SIZES
        .into_iter()
        .chain([192])
        .flat_map(|size| [size - 1, size, size + 1])
        .collect::<BTreeSet<_>>();
    let mut plays = Vec::new();
    for &length in &lengths {
        let hashes = (0..length.div_ceil(SOURCE_BLOCK_SIZE))
            .map(|index| 10 + index as u64)
            .collect::<Vec<_>>();
        let mut extra_hashes = hashes.clone();
        extra_hashes.extend([999, 1000]);
        // Each source trace object is a namespace, so sharing must be tested
        // across requests within that object, never across JSONL objects.
        plays.push(weka_requests(
            &format!("length-{length}"),
            vec![
                weka_request(length, &hashes),
                weka_request(length, &hashes),
                weka_request(length, &extra_hashes),
                weka_request(length, &[10]),
                weka_request(length, &[10]),
            ],
        ));
    }
    write_weka(&source, &plays);
    let direct = load_weka_agentic_graph(&source, Some(SOURCE_BLOCK_SIZE)).unwrap();
    let (summary, rows) = load_weka_agentic_rows(&source).unwrap();
    let materialized = directory.path().join("materialized-v2.jsonl");
    let mut jsonl = serde_json::to_string(&summary.header).unwrap() + "\n";
    for row in &rows {
        jsonl.push_str(&serde_json::to_string(row).unwrap());
        jsonl.push('\n');
    }
    std::fs::write(&materialized, jsonl).unwrap();
    let reparsed = load_agentic_mooncake(&materialized, SOURCE_BLOCK_SIZE).unwrap();
    assert_eq!(direct.identity(), reparsed.identity());
    assert_eq!(direct.nodes(), reparsed.nodes());

    for &length in &lengths {
        let play = format!("length-{length}");
        let a = weka_node(&direct, &play, 0);
        let b = weka_node(&direct, &play, 1);
        let extra = weka_node(&direct, &play, 2);
        let missing_a = weka_node(&direct, &play, 3);
        let missing_b = weka_node(&direct, &play, 4);
        let full_source_units = length / SOURCE_BLOCK_SIZE;
        assert_eq!(a.hash_ids().len(), length.div_ceil(SOURCE_BLOCK_SIZE));
        assert_eq!(extra.hash_ids().len(), a.hash_ids().len());
        assert_eq!(
            &a.hash_ids()[..full_source_units],
            &b.hash_ids()[..full_source_units]
        );
        assert_eq!(
            &a.hash_ids()[..full_source_units],
            &extra.hash_ids()[..full_source_units]
        );
        if full_source_units > 0 {
            assert_eq!(missing_a.hash_ids()[0], a.hash_ids()[0]);
        }
        for index in 1..full_source_units {
            assert_ne!(missing_a.hash_ids()[index], missing_b.hash_ids()[index]);
            assert_ne!(missing_a.hash_ids()[index], a.hash_ids()[index]);
        }
        if !length.is_multiple_of(SOURCE_BLOCK_SIZE) {
            // A supplied hash never grants sharing to an incomplete source
            // unit, even when smaller engine blocks fit inside that tail.
            assert_ne!(a.hash_ids().last(), b.hash_ids().last());
            assert_ne!(a.hash_ids().last(), extra.hash_ids().last());
            assert_ne!(missing_a.hash_ids().last(), missing_b.hash_ids().last());
        }
    }

    for block_size in ENGINE_BLOCK_SIZES {
        let direct_ready = routed_requests(&direct, block_size);
        let reparsed_ready = routed_requests(&reparsed, block_size);
        for node in direct.nodes() {
            let ready = &direct_ready[node.request_id()];
            let materialized_ready = &reparsed_ready[node.request_id()];
            let tokens = direct
                .prompt_materializer()
                .materialize_prefix(node, node.input_length())
                .unwrap();
            assert_eq!(ready.request.tokens, tokens);
            assert_eq!(ready.request.tokens, materialized_ready.request.tokens);
            let native = engine_hashes(&tokens, block_size);
            assert_eq!(
                native.local_block_hashes.len(),
                node.input_length() / block_size
            );
            assert_eq!(ready.replay_hashes.as_ref(), Some(&native));
            assert_eq!(ready.replay_hashes, materialized_ready.replay_hashes);
            assert_eq!(
                direct
                    .prompt_materializer()
                    .replay_hashes(node, node.input_length(), block_size)
                    .unwrap(),
                native
            );
        }
    }
}

#[test]
fn weka_equal_raw_hashes_in_different_namespaces_do_not_share_tokens() {
    let directory = tempdir().unwrap();
    let plays = [weka_play("profile", 257, &[10, 20, 30, 40, 50])];
    write_weka(&directory.path().join("a.jsonl"), &plays);
    write_weka(&directory.path().join("b.jsonl"), &plays);
    let graph = load_weka_agentic_graph(directory.path(), Some(SOURCE_BLOCK_SIZE)).unwrap();
    assert_eq!(graph.node_count(), 2);
    let a = &graph.nodes()[0];
    let b = &graph.nodes()[1];
    for (left, right) in a.hash_ids().iter().zip(b.hash_ids()) {
        assert_ne!(left, right);
    }
    let a_tokens = graph
        .prompt_materializer()
        .materialize_prefix(a, 257)
        .unwrap();
    let b_tokens = graph
        .prompt_materializer()
        .materialize_prefix(b, 257)
        .unwrap();
    assert!(
        a_tokens
            .iter()
            .zip(&b_tokens)
            .all(|(left, right)| left != right)
    );
    for block_size in ENGINE_BLOCK_SIZES {
        let a_hashes = engine_hashes(&a_tokens, block_size);
        let b_hashes = engine_hashes(&b_tokens, block_size);
        assert_ne!(a_hashes.sequence_hashes[0], b_hashes.sequence_hashes[0]);
    }
}

#[test]
fn primer_prefix_preserves_original_full_units_and_private_tail_identity() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.jsonl");
    write_weka(
        &source,
        &[weka_requests(
            "profile-and-primer",
            vec![
                weka_request(223, &[10, 20, 30, 40]),
                weka_request(80, &[10, 20]),
            ],
        )],
    );
    let graph = load_weka_agentic_graph(&source, Some(SOURCE_BLOCK_SIZE)).unwrap();
    let cloned = graph.clone();
    assert!(Arc::ptr_eq(
        graph.prompt_materializer(),
        cloned.prompt_materializer()
    ));
    let profile = weka_node(&graph, "profile-and-primer", 0);
    let full = graph
        .prompt_materializer()
        .materialize_prefix(profile, 223)
        .unwrap();
    let shortened = weka_node(&graph, "profile-and-primer", 1);
    let renormalized = graph
        .prompt_materializer()
        .materialize_prefix(shortened, 80)
        .unwrap();
    assert_eq!(&renormalized[..64], &full[..64]);
    assert_ne!(&renormalized[64..], &full[64..80]);

    for length in [
        0, 1, 63, 64, 65, 80, 95, 96, 127, 128, 129, 192, 193, 207, 222, 223,
    ] {
        let primer = cloned
            .prompt_materializer()
            .materialize_prefix(profile, length)
            .unwrap();
        assert_eq!(primer, full[..length], "prefix length {length}");
        for block_size in ENGINE_BLOCK_SIZES {
            assert_eq!(
                cloned
                    .prompt_materializer()
                    .replay_hashes(profile, length, block_size)
                    .unwrap(),
                engine_hashes(&full[..length], block_size)
            );
        }
    }
    assert!(
        graph
            .prompt_materializer()
            .materialize_prefix(profile, 224)
            .is_err()
    );
}

#[test]
fn unrelated_request_order_cannot_renumber_shared_source_identities() {
    let graphs = ["a-unrelated", "z-unrelated"].map(|unrelated_id| {
        canonical_graph(vec![
            canonical_row(unrelated_id, 64, &[99]),
            canonical_row("m-profile", 256, &[30, 10, 20, 40]),
        ])
    });
    // Validation sorts nodes by request ID. Renaming only the unrelated
    // request moves its source hash ahead of/behind the unchanged profile.
    assert_ne!(
        graphs[0].nodes()[0].request_id(),
        graphs[1].nodes()[0].request_id()
    );
    let source_ids = |graph: &ValidatedAgenticGraph| {
        graph
            .nodes()
            .iter()
            .flat_map(|node| node.hash_ids().iter().copied())
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(source_ids(&graphs[0]), source_ids(&graphs[1]));
    for block_size in ENGINE_BLOCK_SIZES {
        let left = routed_requests(&graphs[0], block_size);
        let right = routed_requests(&graphs[1], block_size);
        assert_eq!(
            left["m-profile"].request.tokens,
            right["m-profile"].request.tokens
        );
        assert_eq!(
            left["m-profile"].replay_hashes,
            right["m-profile"].replay_hashes
        );
    }
}

fn canonical_graph(rows: Vec<AgenticMooncakeRow>) -> ValidatedAgenticGraph {
    AgenticTrace::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: SOURCE_BLOCK_SIZE,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored-test".into(),
                digest: "aic-1889-prefix-fixture".into(),
            },
        },
        rows,
    )
    .unwrap()
}

fn canonical_row(id: &str, input_length: usize, hashes: &[u64]) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: id.into(),
        play_id: id.into(),
        session_id: id.into(),
        model: "model".into(),
        input_length: Some(input_length),
        output_length: Some(1),
        hash_ids: Some(hashes.to_vec()),
        ..Default::default()
    }
}

#[test]
fn rolling_hashes_remember_divergence_when_later_local_blocks_match() {
    let graph = canonical_graph(vec![
        canonical_row("a", 384, &[10, 20, 30, 40, 50, 60]),
        canonical_row("b", 384, &[10, 99, 30, 40, 50, 60]),
    ]);
    for block_size in [16, 32, 48, 64, 96, 128] {
        let ready = routed_requests(&graph, block_size);
        let a = ready["a"].replay_hashes.as_ref().unwrap();
        let b = ready["b"].replay_hashes.as_ref().unwrap();
        let shared_prefix_blocks = SOURCE_BLOCK_SIZE / block_size;
        assert_eq!(
            &a.sequence_hashes[..shared_prefix_blocks],
            &b.sequence_hashes[..shared_prefix_blocks]
        );
        for (left, right) in a.sequence_hashes[shared_prefix_blocks..]
            .iter()
            .zip(&b.sequence_hashes[shared_prefix_blocks..])
        {
            assert_ne!(left, right);
        }
        assert_eq!(a.local_block_hashes.last(), b.local_block_hashes.last());
        assert_ne!(a.sequence_hashes.last(), b.sequence_hashes.last());
    }
}

#[test]
fn weka_normalized_source_and_token_ids_have_literal_goldens() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("golden.json");
    write_weka(
        &source,
        &[weka_play("golden", 257, &[10, 20, 30, 40, 50, 999])],
    );
    let graph = load_weka_agentic_graph(&source, Some(SOURCE_BLOCK_SIZE)).unwrap();
    let node = &graph.nodes()[0];
    let tokens = graph
        .prompt_materializer()
        .materialize_prefix(node, 257)
        .unwrap();
    assert_eq!(
        node.hash_ids(),
        &[
            8068343974007987641,
            3276323911311587801,
            11787140084014995466,
            3679380697349824404,
            13520823821012705895,
        ]
    );
    let mut expected_tokens = [vec![2; 64], vec![0; 64], vec![3; 64], vec![1; 64]].concat();
    expected_tokens.push(4);
    assert_eq!(tokens, expected_tokens);
    let expected_hashes = ReplayRequestHashes {
        local_block_hashes: vec![
            2439962053643786207,
            15480293642169978529,
            15232596819325015918,
            16951273711404654616,
        ],
        sequence_hashes: vec![
            2439962053643786207,
            13979423935614378478,
            11986792261107530170,
            16924757325640419502,
        ],
    };
    assert_eq!(engine_hashes(&tokens, 64), expected_hashes);
    let routed = routed_requests(&graph, 64);
    assert_eq!(routed[node.request_id()].request.tokens, expected_tokens);
    assert_eq!(
        routed[node.request_id()].replay_hashes.as_ref(),
        Some(&expected_hashes)
    );
}

#[test]
fn sorted_u64_source_ids_have_literal_token_and_engine_hash_goldens() {
    // High source IDs deliberately share low 32 bits with other IDs. Truncating
    // source IDs or assigning IDs in traversal order cannot pass this vector.
    let graph = canonical_graph(vec![canonical_row(
        "golden",
        257,
        &[u64::MAX - 1, 0, 1_u64 << 32, u64::MAX, 7],
    )]);
    let mut expected_tokens = [vec![3; 64], vec![0; 64], vec![2; 64], vec![4; 64]].concat();
    expected_tokens.push(1);
    let node = &graph.nodes()[0];
    assert_eq!(
        graph
            .prompt_materializer()
            .materialize_prefix(node, 257)
            .unwrap(),
        expected_tokens
    );

    // Fixed values for the authored token vector above: little-endian u32
    // blocks, XXH3 seed 1337, then a little-endian (parent, local) hash chain.
    // These literals are an oracle independent of either Rust helper.
    let goldens: &[(usize, &[u64], &[u64])] = &[
        (
            16,
            &[
                18442881532971542694,
                18442881532971542694,
                18442881532971542694,
                18442881532971542694,
                5523182284766269584,
                5523182284766269584,
                5523182284766269584,
                5523182284766269584,
                4954898952464910565,
                4954898952464910565,
                4954898952464910565,
                4954898952464910565,
                18402572742196698181,
                18402572742196698181,
                18402572742196698181,
                18402572742196698181,
            ],
            &[
                18442881532971542694,
                3110247045411300890,
                17446551311482910668,
                1462161110062356123,
                10950884676873469705,
                15199822948515376398,
                18201976680066987796,
                2973487274555673469,
                7873989502860313589,
                1926401905964609976,
                14847415723312173268,
                17377089510758635973,
                11747487590084227061,
                4023604402287706361,
                4011548761091022455,
                8026677643285169965,
            ],
        ),
        (
            32,
            &[
                2671291174597333161,
                2671291174597333161,
                8078176463474809903,
                8078176463474809903,
                12481126450064483879,
                12481126450064483879,
                11107320821997019568,
                11107320821997019568,
            ],
            &[
                2671291174597333161,
                1380531233511415844,
                15625092046538750603,
                5423220605135736781,
                3470886858311107834,
                9723483233950824890,
                16563788231771075685,
                7367871862247061576,
            ],
        ),
        (
            48,
            &[
                10861956436164017793,
                2265553212248219955,
                16943279169571679943,
                9744416516818080641,
                1223047621970469029,
            ],
            &[
                10861956436164017793,
                16096818866154992370,
                1853191128699898713,
                10986734978503536101,
                5629142904534639979,
            ],
        ),
        (
            64,
            &[
                15232596819325015918,
                15480293642169978529,
                2439962053643786207,
                15517538317340902020,
            ],
            &[
                15232596819325015918,
                14048058817150525812,
                7687217209076540820,
                1312705216624093154,
            ],
        ),
        (
            96,
            &[11781536417453921607, 11126741662045547019],
            &[11781536417453921607, 7247272651566549697],
        ),
        (
            128,
            &[2422809868827000573, 16118892911259521380],
            &[2422809868827000573, 13183720575708183136],
        ),
        (256, &[1845043586253682500], &[1845043586253682500]),
    ];
    for &(block_size, local, sequence) in goldens {
        let ready = routed_requests(&graph, block_size);
        assert_eq!(ready["golden"].request.tokens, expected_tokens);
        let native = engine_hashes(&expected_tokens, block_size);
        assert_eq!(
            native.local_block_hashes, local,
            "engine block size {block_size}"
        );
        assert_eq!(
            native.sequence_hashes, sequence,
            "engine block size {block_size}"
        );
        assert_eq!(ready["golden"].replay_hashes.as_ref(), Some(&native));
    }
}
