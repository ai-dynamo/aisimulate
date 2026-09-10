# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from collector.sglang.dsv41_forward_results import BOUNDARY, aggregate_forward_results, forward_admission_report
from collector.sglang.dsv41_workloads import freeze_workloads

pytestmark = pytest.mark.unit


@pytest.fixture
def attempt(tmp_path):
    plan = freeze_workloads(
        {"schema_version": 3, "prefill": [], "decode": [{"batch_size": 2, "total_kv_read_tokens": 256}]}
    )
    sources = {"model.py": "a" * 64}
    identity = {
        "source_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "config_sha256": "b" * 64,
        "runtime_digest": "sha256:" + "c" * 64,
        "execution_profile": "full",
    }
    contract = {
        "mode": "native_benchmark_forward",
        "component_recorder": False,
        "timing_boundary": BOUNDARY,
        "warmup": 1,
        "iterations": 3,
        "native_cli_args": [
            "--disable-custom-all-reduce",
            "--enforce-disable-flashinfer-allreduce-fusion",
            "--disable-shared-experts-fusion",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ],
    }
    for name, value in {
        "workload-plan": plan,
        "source_hashes": sources,
        "input_provenance": {**identity, "source": "tokenizer_text", "text_sha256": "d" * 64},
        "execution-contract": contract,
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    (tmp_path / "COMPLETE").write_text("native workload collection completed\n")
    for rank in range(4):
        rows = [
            {
                **plan["cases"][0],
                **identity,
                "invocation": sample + 1,
                "sample": sample,
                "tp_rank": rank,
                "real_kv": True,
                "finite_logits": True,
                "component_recorder": False,
                "used_cuda_graph": False,
                "timing_boundary": BOUNDARY,
                "native_benchmark_forward_ms": sample + rank / 10,
            }
            for sample in range(1, 4)
        ]
        (tmp_path / f"forward-rank-{rank}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return tmp_path


def test_independent_forward_preserves_rank_max_repetitions(attempt):
    result = aggregate_forward_results(attempt)
    assert result["case_count"] == 1
    assert result["cases"][0]["rank_max_ms"] == [1.3, 2.3, 3.3]
    assert result["cases"][0]["median_ms"] == 2.3
    assert result["cases"][0]["canonical_past_kv"] == 128
    assert result["cases"][0]["native_inclusive_kv"] == 129


@pytest.mark.parametrize(
    "failure", ["missing_rank", "missing_sample", "duplicate_rank", "wrong_axis", "mixed_source", "module"]
)
def test_incomplete_or_contaminated_forward_attempt_is_rejected(attempt, failure):
    path = attempt / "forward-rank-3.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing_rank":
        path.unlink()
    elif failure == "module":
        (attempt / "rank-0.jsonl").write_text("{}\n")
    else:
        if failure == "missing_sample":
            rows.pop()
        elif failure == "duplicate_rank":
            rows.append(rows[0])
        elif failure == "wrong_axis":
            rows[0]["native_inclusive_kv"] -= 1
        elif failure == "mixed_source":
            rows[0]["source_sha256"] = "f" * 64
        path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError):
        aggregate_forward_results(attempt)


def test_rejected_forward_preserves_failure_and_missing_rank_inventory(attempt):
    (attempt / "forward-rank-3.jsonl").unlink()
    failed = {"case_id": "decode-0000", "tp_rank": 3, "status": "failed", "error_type": "RuntimeError"}
    (attempt / "workloads-rank-3.jsonl").write_text(json.dumps(failed) + "\n")
    report = forward_admission_report(attempt)
    assert report["status"] == "rejected" and report["complete"] is False
    assert report["missing_rank_files"] == ["forward-rank-3.jsonl"]
    assert report["failed_workloads"] == [failed]
    assert report["missing_observations"] == [
        {"case_id": "decode-0000", "sample": sample, "missing_ranks": [3]} for sample in range(1, 4)
    ]
    assert "cases" not in report
