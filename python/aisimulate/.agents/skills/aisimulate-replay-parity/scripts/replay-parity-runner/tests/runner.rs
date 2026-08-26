// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn fixture() -> (TempDir, PathBuf, PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let trace = directory.path().join("trace.jsonl");
    let trace_bytes = concat!(
        "{\"timestamp\":0,\"input_length\":4,\"output_length\":2,\"hash_ids\":[1]}\n",
        "{\"timestamp\":1,\"input_length\":4,\"output_length\":2,\"hash_ids\":[1]}\n"
    )
    .as_bytes();
    fs::write(&trace, trace_bytes).unwrap();
    let config = directory.path().join("config.json");
    let document = json!({
        "row_name": "test-vllm-aggregated",
        "trace_file": trace,
        "trace_sha256": sha256(trace_bytes),
        "trace_rows": 2,
        "trace_block_size": 4,
        "arrival_speedup_ratio": 1.0,
        "trace_upstream_repository": "https://example.invalid/repository",
        "trace_upstream_commit": "0000000000000000000000000000000000000000",
        "trace_upstream_path": "trace.jsonl",
        "trace_full_sha256": sha256(trace_bytes),
        "trace_full_rows": 2,
        "trace_slice_start": 0,
        "iterations": 1,
        "capture_canonical": true,
        "write_full_report": false,
        "metadata": {},
        "expected": {
            "completed_requests": 2,
            "requested_output_tokens": 4,
            "total_output_tokens": 4,
            "all_requests_full_output": true,
            "requests_with_short_output": 0,
            "decode_workers": [0]
        },
        "spec": {
            "version": 1,
            "topology": {
                "kind": "aggregated",
                "workers": {"initial_workers": 1, "startup_delay_ms": 0.0}
            },
            "engine": {
                "dp_size": 1,
                "tensor_parallel_size": 1,
                "rank": {
                    "backend": "vllm",
                    "num_gpu_blocks": 16,
                    "block_size": 4,
                    "max_num_seqs": 4,
                    "max_num_batched_tokens": 64,
                    "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                }
            },
            "adapters": {
                "placement": {"provider": "round_robin", "config": null},
                "scaling": {"provider": "none", "config": null}
            },
            "record_per_request": true,
            "requests": []
        }
    });
    fs::write(&config, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    (directory, config, trace)
}

fn run(config: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aisimulate-replay-parity-runner"))
        .arg(config)
        .arg(output)
        .output()
        .unwrap()
}

fn mutate_config(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

#[test]
fn two_fresh_processes_emit_identical_canonical_output() {
    let (directory, config, _) = fixture();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    assert!(run(&config, &first).status.success());
    assert!(run(&config, &second).status.success());
    assert_eq!(
        fs::read(first.join("canonical.jsonl")).unwrap(),
        fs::read(second.join("canonical.jsonl")).unwrap()
    );
}

#[test]
fn rejects_trace_checksum_mismatch() {
    let (directory, config, _) = fixture();
    mutate_config(&config, |value| {
        value["trace_sha256"] = json!("0".repeat(64))
    });
    let result = run(&config, &directory.path().join("output"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("trace SHA-256 mismatch"));
}

#[test]
fn rejects_user_supplied_source_revision() {
    let (directory, config, _) = fixture();
    mutate_config(&config, |value| {
        value["source_revision"] = json!("0".repeat(40))
    });
    let result = run(&config, &directory.path().join("output"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown field `source_revision`"));
}

#[test]
fn rejects_trace_row_count_mismatch() {
    let (directory, config, _) = fixture();
    mutate_config(&config, |value| {
        value["trace_rows"] = json!(3);
        value["trace_full_rows"] = json!(3);
    });
    let result = run(&config, &directory.path().join("output"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("trace row-count mismatch"));
}

#[test]
fn rejects_nonempty_output_directory() {
    let (directory, config, _) = fixture();
    let output = directory.path().join("output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("owned-by-user"), b"preserve").unwrap();
    let result = run(&config, &output);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("is not empty"));
}

#[test]
fn rejects_non_round_robin_or_scaling_composition() {
    for (path, value, expected) in [
        (
            "/spec/adapters/placement/provider",
            "kv_router",
            "requires round_robin placement",
        ),
        (
            "/spec/adapters/scaling/provider",
            "planner",
            "requires scaling provider none",
        ),
    ] {
        let (directory, config, _) = fixture();
        mutate_config(&config, |document| {
            *document.pointer_mut(path).unwrap() = json!(value)
        });
        let result = run(&config, &directory.path().join("output"));
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains(expected));
    }
}

#[test]
fn rejects_golden_expectation_mismatch() {
    let (directory, config, _) = fixture();
    mutate_config(&config, |value| {
        value["expected"]["completed_requests"] = json!(999)
    });
    let result = run(&config, &directory.path().join("output"));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("golden expectation mismatch"));
}
