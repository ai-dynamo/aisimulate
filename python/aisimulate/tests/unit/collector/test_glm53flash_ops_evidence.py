# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored CPU receipts exercise integrity; no fixture is GPU perf evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config
from collector import glm53flash_validation as evidence
from collector.fpm_forward.sglang_artifact import TELEMETRY_POLICY
from collector.glm53flash_contract import (
    BACKENDS,
    CHECKPOINTS,
    aggregate_rank_records,
    canonical_json,
    sha256_json,
    write_parquet,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def authored_graph_source(monkeypatch):
    # These integrity tests use one authored op, not a GPU/model dataset. The
    # production manifest itself is covered by the public model/getter tests.
    geometry = canonical_json({"backend": "sglang", "checkpoint_format": "fp8", "tp_size": 2, "is_context": False})
    monkeypatch.setattr(
        evidence,
        "build_model_manifest",
        lambda *_: {
            "phases": {"generation": [{"name": "attention_0", "component": "attention", "geometry": geometry}]}
        },
    )


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def put_lines(path, rows):
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def native_fixture(tmp_path, role="holdout"):
    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity

    root = tmp_path / "native"
    root.mkdir()
    pins = json.loads(
        (
            Path(evidence.__file__).parent / "fpm_forward/runtime/glm53flash_sglang/runtime-source-sha256.json"
        ).read_text()
    )
    put(
        root / "runtime-preflight.json",
        {"status": "passed", "backend": "sglang", "backend_version": "0.5.20", "sources": pins},
    )
    point = {
        "benchmark_id": 1,
        "point_type": "decode",
        "batch_size": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2,
    }
    requests = {"request_set": "authored-cpu-fixture", "dataset_role": role, "corpus_sha256": "a" * 64, "requests": {}}
    for rep in range(15):
        requests["requests"][f"request-{rep}"] = {
            "benchmark_id": 1,
            "repetition": rep,
            "sampling_role": "warmup" if rep < 5 else "measurement",
            "target_phase": "generation",
            "target_query": 1,
            "target_prefix": 2,
            "target_batch_size": 1,
        }
    put(root / "requests.json", requests)
    identity = {
        "backend": "sglang",
        "backend_version": BACKENDS["sglang"][0],
        "backend_revision": BACKENDS["sglang"][1],
        "checkpoint_revision": CHECKPOINTS["fp8"][1],
        "config_sha256": sha256_json(_load_pre_downloaded_hf_config(CHECKPOINTS["fp8"][0])),
        "source_sha256": sha256_json(pins),
        "runtime_digest": "sha256:" + "b" * 64,
        "run_id": "authored-unit-run",
        "execution_identity": dict(
            zip(
                EXECUTION_COLUMNS,
                execution_identity(
                    _load_pre_downloaded_hf_config(CHECKPOINTS["fp8"][0]), backend="sglang", input_modality="text"
                ),
                strict=True,
            )
        ),
        "telemetry_policy": TELEMETRY_POLICY,
        "context_policy": {
            "measured_context_limit": 131072,
            "runtime_context_length": 131079,
            "native_admission_headroom": 7,
        },
    }
    put(root / "sglang-provenance.json", identity)
    native_config = {
        "model_path": "/models/authored-cpu-fixture",
        "revision": CHECKPOINTS["fp8"][1],
        "tp_size": 2,
        "pp_size": 1,
        "dp_size": 1,
        "ep_size": 1,
        "attn_cp_size": 1,
        "nnodes": 1,
        "context_length": 131079,
        "kv_cache_dtype": "fp8_e4m3",
        "disable_radix_cache": True,
        "chunked_prefill_size": 8192,
        "cuda_graph_config": {"prefill": {"backend": "disabled"}, "decode": {"backend": "disabled"}},
    }
    put(root / "sglang-declared-config.json", native_config)
    put(root / "sglang-resolved-config.json", native_config)
    layout = {
        "admitted": True,
        "logical_kv_dtype": "torch.float8_e4m3fn",
        "groups": {
            name: [{"dtype": dtype, "shape": [34 if name.startswith("kda_") else 11, 2, 4]}]
            for name, dtype in {
                "kda_conv": "torch.bfloat16",
                "kda_temporal": "torch.float32",
                "mla_latent": "torch.float8_e4m3fn",
                "pooled_index_packed": "torch.uint8",
                "index_tail_key": "torch.bfloat16",
                "index_tail_score": "torch.bfloat16",
            }.items()
        },
    }
    all_records, all_modules = {}, {}
    geometry = canonical_json({"backend": "sglang", "checkpoint_format": "fp8", "tp_size": 2, "is_context": False})
    for rank in range(2):
        put(root / f"state-layout-rank-{rank}.json", layout)
        records, modules = [], []
        for rep in range(15):
            rid = f"request-{rep}"
            sampling_role = "warmup" if rep < 5 else "measurement"
            for step in (0, 1):
                query, prefix, tokens, sample = (2, 0, [4, 5], 7) if not step else (1, 2, [7], 9)
                fid = f"rank-{rank}/forward-{rep * 2 + step}"
                gpu_ms = 999.0 if rep < 5 else (1.0 if rank == 0 else 3.0) if rep < 10 else (7.0 if rank == 0 else 5.0)
                row = {
                    **identity,
                    "forward_id": fid,
                    "invocation": rep * 2 + step,
                    "tp_rank": rank,
                    "allocated_fake_tokens": 0,
                    "state_protocol": "glm53flash_same_request_real_hybrid_v1",
                    "gpu_completed": True,
                    "state_layout_admitted": True,
                    "state_layout_sha256": hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
                    "used_cuda_graph": False,
                    "runtime_mode": "NONE",
                    "ops_instrumented": role == "calibration",
                    "batch_size": 1,
                    "request_ids": [rid],
                    "query_lengths": [query],
                    "prefix_lengths": [prefix],
                    "total_new_tokens": query,
                    "total_past_kv_tokens": prefix,
                    "stage": "seed" if not step else "measure",
                    "phase": "context" if not step else "generation",
                    "benchmark_id": 1,
                    "repetition": rep,
                    "sampling_role": sampling_role,
                    "dataset_role": role,
                    "request_set": requests["request_set"],
                    "corpus_sha256": requests["corpus_sha256"],
                    "native_forward_ms": 123456.0,
                    "whole_forward_gpu_ms": gpu_ms,
                    "whole_forward_boundary": evidence.BOUNDARY,
                    "requests": [
                        {
                            "request_id": rid,
                            "native_query_token_ids": tokens,
                            "prompt_token_ids": [4, 5],
                            "previous_forward_id": f"rank-{rank}/forward-{rep * 2}" if step else None,
                            "same_request_real_prefix": True,
                            "computed_tokens_before": prefix,
                            "computed_tokens_after": prefix + query,
                            "input_tokens_sha256": sha256_json([4, 5, 7] if step else [4, 5]),
                            "sampled_token_id": sample,
                        }
                    ],
                }
                records.append(row)
                if step:
                    modules.append(
                        {
                            **identity,
                            "component": "attention",
                            "geometry": geometry,
                            "batch_size": 1,
                            "prefix": 0,
                            "x": 2,
                            "latency": gpu_ms,
                            "sample_count": 10,
                            "measurement_scope": "local_compute",
                            "kv_seed_regime": "real_kv",
                            "dispatch_fingerprint": "",
                            "used_cuda_graph": False,
                            "kernel_source": "authored.fixture.native.forward",
                            "state_mode": "decode",
                            "dataset_role": role,
                            "request_set": requests["request_set"],
                            "corpus_sha256": requests["corpus_sha256"],
                            "evidence_sha256": "e" * 64,
                            "name": "attention_0",
                            "phase": "generation",
                            "sample": rep,
                            "invocation": rep * 2 + step,
                            "tp_rank": rank,
                            "stage": "measure",
                            "benchmark_id": 1,
                            "repetition": rep,
                            "sampling_role": sampling_role,
                            "request_ids": [rid],
                            "history_ids": [f"rank-{rank}/forward-{rep * 2}"],
                        }
                    )
        put_lines(root / f"forward-rank-{rank}.jsonl", records)
        put_lines(root / f"rank-{rank}.jsonl", modules)
        all_records[rank], all_modules[rank] = records, modules
    manifest = {
        "backend": "sglang",
        "checkpoint_revision": CHECKPOINTS["fp8"][1],
        "phases": {"generation": [{"name": "attention_0", "component": "attention", "geometry": geometry}]},
    }
    put(root / "manifest.json", manifest)
    put(root / "points.json", {"schema_version": 3, "decode": [point], "prefill": []})
    put(root / "provenance.json", identity)
    put(root / "command.json", {"argv": ["authored-unit-fixture-only"]})
    if role == "calibration":
        evidence.freeze_evidence(root, 2, manifest)
    run = {
        "spec": {"raw_root": str(root)},
        "key": ("sglang", "fp8", 2, "decode"),
        "role": role,
        "corpus": "a" * 64,
        "points": [point],
        "plan": {"sha256": "d" * 64},
    }
    return run, root, all_records, all_modules, manifest


def test_whole_gpu_truth_is_rank_max_then_median_and_never_host_fpm(tmp_path):
    run, _, _, _, _ = native_fixture(tmp_path)
    result = evidence.load_native(run, tmp_path)
    assert result["values"] == {1: 5.0}
    assert result["timing_boundary"] == "embedding_to_logits_gpu_v1"
    assert len(result["request_ids"]) == 15


@pytest.mark.parametrize("field", ["run_id", "context_policy", "execution_identity"])
def test_ops_rejects_consistent_cross_rank_trace_from_another_execution(tmp_path, field):
    run, root, records, _, _ = native_fixture(tmp_path)
    for rank, rows in records.items():
        for row in rows:
            row[field] = "another-run" if field == "run_id" else {"unrelated": "identity"}
        put_lines(root / f"forward-rank-{rank}.jsonl", rows)
    with pytest.raises(ValueError, match="raw forward execution"):
        evidence.load_native(run, tmp_path)


def test_ops_rejects_resolved_graph_enablement_even_when_trace_says_eager(tmp_path):
    run, root, _, _, _ = native_fixture(tmp_path)
    config = json.loads((root / "sglang-resolved-config.json").read_bytes())
    config["cuda_graph_config"]["decode"]["backend"] = "full"
    put(root / "sglang-resolved-config.json", config)
    with pytest.raises(ValueError, match="both native SGLang graph phases disabled"):
        evidence.load_native(run, tmp_path)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_gpu",
        "wrong_boundary",
        "instrumented",
        "missing_instrumentation_flag",
        "changed_source",
        "wrong_token",
        "missing_seed",
    ],
)
def test_holdout_rejects_native_evidence_corruption(tmp_path, tamper):
    run, root, records, _, _ = native_fixture(tmp_path)
    row = records[1][11]
    if tamper == "missing_gpu":
        row.pop("whole_forward_gpu_ms")
    elif tamper == "wrong_boundary":
        row["whole_forward_boundary"] = "vllm_native_scheduler_output_interval"
    elif tamper == "instrumented":
        row["ops_instrumented"] = True
    elif tamper == "missing_instrumentation_flag":
        row.pop("ops_instrumented")
    elif tamper == "changed_source":
        audit = json.loads((root / "runtime-preflight.json").read_text())
        audit["sources"][next(iter(audit["sources"]))] = "f" * 64
        put(root / "runtime-preflight.json", audit)
    elif tamper == "wrong_token":
        row["requests"][0]["native_query_token_ids"] = [88]
        row["requests"][0]["input_tokens_sha256"] = sha256_json([4, 5, 88])
    else:
        records[1].pop(10)
    put_lines(root / "forward-rank-1.jsonl", records[1])
    with pytest.raises(ValueError):
        evidence.load_native(run, tmp_path)


def test_calibration_evidence_hash_detects_raw_file_changes(tmp_path):
    run, root, records, _, _ = native_fixture(tmp_path, "calibration")
    assert evidence.load_native(run, tmp_path)["values"] == {}
    records[1][11]["native_forward_ms"] += 1
    put_lines(root / "forward-rank-1.jsonl", records[1])
    with pytest.raises(ValueError, match="changed after calibration"):
        evidence.load_native(run, tmp_path)


def test_consumer_rows_are_reaggregated_from_original_module_measurements(tmp_path):
    run, root, _, _, manifest = native_fixture(tmp_path, "calibration")
    native = evidence.load_native(run, tmp_path)
    rows = aggregate_rank_records(
        [root / "rank-0.jsonl", root / "rank-1.jsonl"],
        2,
        manifest,
        evidence_sha256=evidence.file_sha(root / "calibration-evidence.json"),
    )
    parquet = tmp_path / "glm53flash_module_perf.parquet"
    write_parquet(rows, parquet)
    assert evidence.bind_calibration([parquet], run, native)["rows"] == 1
    rows[0]["latency"] += 0.1
    write_parquet(rows, parquet)
    with pytest.raises(ValueError, match="differ from their native calibration"):
        evidence.bind_calibration([parquet], run, native)


def test_module_rows_cannot_borrow_another_request_set_inside_hashed_bundle(tmp_path):
    run, root, _, modules, manifest = native_fixture(tmp_path, "calibration")
    for rank in modules:
        for row in modules[rank]:
            row["request_set"] = "unrelated-native-run"
            row["corpus_sha256"] = "f" * 64
            row["request_ids"] = ["unrelated-request"]
        put_lines(root / f"rank-{rank}.jsonl", modules[rank])
    evidence.freeze_evidence(root, 2, manifest)
    with pytest.raises(ValueError):
        native = evidence.load_native(run, tmp_path)
        rows = aggregate_rank_records(
            [root / "rank-0.jsonl", root / "rank-1.jsonl"],
            2,
            manifest,
            evidence_sha256=evidence.file_sha(root / "calibration-evidence.json"),
        )
        parquet = tmp_path / "glm53flash_module_perf.parquet"
        write_parquet(rows, parquet)
        evidence.bind_calibration([parquet], run, native)


def test_calibration_manifest_cannot_omit_production_graph_occurrences(tmp_path):
    run, root, _, _, manifest = native_fixture(tmp_path, "calibration")
    manifest["phases"]["generation"] = []
    put(root / "manifest.json", manifest)
    evidence.freeze_evidence(root, 2, manifest)
    native = evidence.load_native(run, tmp_path)
    with pytest.raises(ValueError, match="complete production graph"):
        evidence.bind_calibration([], run, native)
