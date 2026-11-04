# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable HF inputs must retain identities, checksums and workload semantics."""

import copy
import hashlib
import json

import pytest
from test_agentx_accuracy import result, silicon_inputs

from scripts.e2e_accuracy.agentx_contract import canonical, validate_catalog
from scripts.e2e_accuracy.agentx_inputs import (
    checked,
    finite_trace,
    load_runs,
    manual_geometry,
    reconstruct,
    safe_path,
)
from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES, validate


def test_thin_catalog_and_relative_bundle_round_trip(tmp_path):
    catalog = silicon_inputs(tmp_path, result())
    assert validate_catalog(catalog) == catalog
    runs = load_runs(tmp_path, catalog)
    assert runs[0]["id"] == result()["id"]
    assert {
        r["id"]
        for r in json.loads(checked(tmp_path, runs[0]["artifacts"]["feature_metadata"]).read_bytes())["features"]
    } == SILICON_FEATURES
    manifest = checked(tmp_path, catalog["runs"][0]["manifest"])
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_runs(tmp_path, catalog)


@pytest.mark.parametrize("path", ["../escape.json", "/absolute", "a//b", "a/./b", "a/../b", "a\\b"])
def test_unsafe_bundle_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, path)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, "link/escape.json")


@pytest.mark.parametrize("child_link", [False, True])
def test_trusted_root_with_symlinked_ancestor(tmp_path, child_link):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    root = alias / "inputs"
    root.mkdir()
    target = root / "bundle"
    target.mkdir()
    if child_link:
        (root / "link").symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path(root, "link/manifest.json")
    else:
        assert safe_path(root, "bundle/manifest.json") == target / "manifest.json"


def test_corpus_filter_preserves_original_bytes_and_order(tmp_path):
    source = tmp_path / "source.jsonl"
    lines = [
        b'{"id":"small", "requests":[{"in":4,"out":2}]}\n',
        b'{"id":"large","requests":[{"in":9,"out":2}]}\n',
    ]
    source.write_bytes(b"".join(lines))
    descriptor = {
        "repository": "org/corpus",
        "revision": "a" * 40,
        "filename": "traces.jsonl",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "derived_sha256": hashlib.sha256(lines[0]).hexdigest(),
        "transformations": [
            {
                "operation": "retain_complete_traces",
                "maximum_tokens": 8,
                "eligible": ["small"],
                "excluded": ["large"],
                "serialization": "original_jsonl_lines",
            }
        ],
    }
    target = tmp_path / "filtered.jsonl"
    reconstruct(descriptor, source, target)
    assert target.read_bytes() == lines[0]
    descriptor["transformations"][0]["eligible"] = ["invented"]
    with pytest.raises(ValueError, match="reviewed workload"):
        reconstruct(descriptor, source, target)
    assert not target.exists()


def test_finite_trace_preserves_token_equality_and_warmup_barrier(tmp_path):
    rows = [
        {
            "request_id": identity,
            "cell_id": "case",
            "lane_id": 0,
            "turn_index": turn,
            "phase": phase,
            "input_token_ids": prompt,
            "output_token_ids": output,
            "dependency_ids": [],
            "think_time_ms": 100,
        }
        for identity, turn, phase, prompt, output in [
            ("warm", -1, "warmup", [80, 90], [70]),
            ("measured", 0, "measured", [80, 90, 70], [60]),
            ("kernel", -2, "kernel_warmup", [1], [2]),
        ]
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b"".join(canonical(r) for r in rows))
    run = {
        "model": "test/model",
        "artifacts": {
            "raw": {
                "path": raw.name,
                "bytes": raw.stat().st_size,
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        },
    }
    target = tmp_path / "trace.jsonl"
    finite_trace(tmp_path, run, target)
    records = [json.loads(line) for line in target.read_text().splitlines()][1:]
    assert {r["request_id"] for r in records} == {"warm", "measured"}
    measured = next(r for r in records if r["request_id"] == "measured")
    warm = next(r for r in records if r["request_id"] == "warm")
    assert measured["hash_ids"] == [80, 90, 70]
    assert warm["output_token_ids"] == [2]
    assert measured["dependencies"] == [
        {
            "request_id": "warm",
            "trigger": "completion",
            "delay_ms": 100,
            "relation": "sequence",
        }
    ]


def test_curated_features_keep_actual_draft_length_and_unknown_scope(tmp_path):
    catalog = silicon_inputs(tmp_path, result())
    run = load_runs(tmp_path, catalog)[0]
    value = json.loads(checked(tmp_path, run["artifacts"]["feature_metadata"]).read_bytes())
    row = next(r for r in value["features"] if r["id"] == "speculation")
    row.update(
        state="enabled",
        parameters=[
            {"name": "method", "value": "dspark"},
            {"name": "draft_length", "value": 7},
        ],
        evidence=[{"sha256": "a" * 64, "pointer": "/runtime"}],
    )
    assert validate(copy.deepcopy(value)) == value
    assert next(r for r in value["features"] if r["id"] == "cache.g2")["parameters"] == [
        {"name": "scope", "value": "unknown"}
    ]
    row["parameters"][0]["value"] = "off"
    with pytest.raises(ValueError, match="not applicable"):
        validate(value)


def test_partial_measurement_snapshot_cannot_pass_public_gate():
    from test_agentx_accuracy import bundle

    from scripts.pages.agentx_pages import validate_snapshot

    snapshot = json.loads(bundle()["summary.json"])["snapshot"]
    snapshot["measurement_campaign_status"] = "partial"
    with pytest.raises(ValueError, match="require a preview"):
        validate_snapshot(snapshot)
    snapshot["preview"] = True
    validate_snapshot(snapshot, preview=True)
    snapshot["measurement_campaign_status"] = "complete"
    with pytest.raises(ValueError, match="require a preview"):
        validate_snapshot(snapshot, preview=True)


@pytest.fixture
def cache_workers():
    groups = [
        {
            "layer_names": [f"layer-{g * 15 + i}" for i in range(15)],
            "spec_type": "FullAttentionSpec" if g == 0 else "MambaSpec",
            "page_size_bytes": 256,
            "spec": {"block_size": 64, **({"mamba_cache_mode": "align"} if g else {})},
        }
        for g in range(4)
    ]
    worker = {
        "vllm_version": "0.31.0",
        "speculative_config": None,
        "kv_transfer_config": None,
        "model": {"model": "Qwen/Qwen3.5-397B-A17B"},
        "parallelism": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
        },
        "scheduler": {
            "async_scheduling": False,
            "enable_chunked_prefill": True,
            "disable_hybrid_kv_cache_manager": False,
        },
        "cache_config": {
            "block_size": 64,
            "mamba_cache_mode": "align",
            "enable_prefix_caching": True,
            "prefix_match_unit": None,
            "use_replayssm": False,
        },
        "prefix_cache_retention_interval": 0,
        "num_blocks": 10,
        "groups": groups,
        "tensors": [
            {"size": 38400, "offset": 0, "layers": g["layer_names"], "layer_stride": 2560, "block_stride": 256}
            for g in groups
        ],
    }
    return [dict(copy.deepcopy(worker), rank=rank) for rank in range(8)]


def test_qwen_geometry_retains_shared_pool_and_reserved_blocks(cache_workers):
    assert manual_geometry(cache_workers) == {
        "block_size": 64,
        "prefix_match_unit": 64,
        "bytes_per_token": 60,
        "prefix_caching": True,
        "capacity": {"type": "fixed", "blocks": 9},
        "state_cache": {"bytes_per_request": 11520},
    }


@pytest.mark.parametrize(
    "drift", ["missing_rank", "duplicate_rank", "out_of_range_rank", "bool_rank", "capacity", "scheduler", "tensor"]
)
def test_qwen_geometry_rejects_rank_or_allocation_drift(cache_workers, drift):
    if drift == "missing_rank":
        cache_workers.pop()
    elif drift == "duplicate_rank":
        cache_workers[-1]["rank"] = 0
    elif drift == "out_of_range_rank":
        cache_workers[-1]["rank"] = 8
    elif drift == "bool_rank":
        cache_workers[0]["rank"] = False
    elif drift == "capacity":
        cache_workers[-1]["num_blocks"] += 1
    elif drift == "scheduler":
        cache_workers[-1]["scheduler"]["max_num_seqs"] = 32
    else:
        cache_workers[-1]["tensors"][0]["size"] += 1
    with pytest.raises(ValueError):
        manual_geometry(cache_workers)
