# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from pathlib import Path

import pytest
from collector.sglang.dsv41_forward_results import BOUNDARY, aggregate_forward_results, forward_admission_report
from collector.sglang.dsv41_workloads import freeze_workloads

pytestmark = pytest.mark.unit


@pytest.fixture
def attempt(tmp_path, request):
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
    for rank in range(getattr(request, "param", 4)):
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
        progress = [
            {
                **plan["cases"][0],
                "case_index": 0,
                "sample": sample,
                "tp_rank": rank,
                "measured": sample >= 1,
                "status": "passed",
            }
            for sample in range(4)
        ]
        (tmp_path / f"workloads-rank-{rank}.jsonl").write_text("\n".join(map(json.dumps, progress)) + "\n")
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


def test_forward_admission_requires_the_recorded_warmup(attempt):
    path = attempt / "workloads-rank-2.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="warmup/progress"):
        aggregate_forward_results(attempt)


@pytest.mark.parametrize("attempt", [8], indirect=True)
def test_forward_report_and_cli_use_configured_tensor_parallel_size(attempt, monkeypatch):
    from collector.sglang.dsv41_forward_results import main

    report = forward_admission_report(attempt, tp_size=8)
    assert report["status"] == "accepted"
    assert report["rank_count"] == 8
    assert report["cases"][0]["rank_max_ms"] == [1.7, 2.7, 3.7]
    output = attempt / "admission.json"
    monkeypatch.setattr(sys, "argv", ["admit", str(attempt), "--output", str(output), "--tp-size", "8"])
    main()
    assert json.loads(output.read_text()) == report

    (attempt / "forward-rank-7.jsonl").unlink()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    rejected = json.loads(output.read_text())
    assert rejected["status"] == "rejected"
    assert rejected["missing_rank_files"] == ["forward-rank-7.jsonl"]
    assert rejected["missing_observations"] == [
        {"case_id": "decode-0000", "sample": sample, "missing_ranks": [7]} for sample in range(1, 4)
    ]


@pytest.mark.parametrize("tp_size", [0, -1])
def test_forward_report_rejects_nonpositive_tensor_parallel_size(attempt, tp_size):
    report = forward_admission_report(attempt, tp_size=tp_size)
    assert report["status"] == "rejected"
    assert report["admission_error"] == "tp_size must be positive"


@pytest.mark.parametrize("failure", ["read", "discovery"])
def test_progress_inventory_io_failure_preserves_primary_admission_error(attempt, monkeypatch, failure):
    (attempt / "COMPLETE").unlink()
    if failure == "read":
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "workloads-rank-2.jsonl":
                raise OSError("progress file disappeared")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
        expected = {
            "file": "workloads-rank-2.jsonl",
            "status": "unreadable_progress_record",
            "error": "progress file disappeared",
        }
    else:
        original = Path.glob

        def glob(path, pattern):
            if pattern == "workloads-rank-*.jsonl":
                raise OSError("progress directory unreadable")
            return original(path, pattern)

        monkeypatch.setattr(Path, "glob", glob)
        expected = {"status": "unreadable_progress_inventory", "error": "progress directory unreadable"}
    report = forward_admission_report(attempt)
    assert report["status"] == "rejected"
    assert report["admission_error"] == "native forward attempt has no completion receipt"
    assert report["failed_workloads"] == [expected]
    assert report["missing_observations"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "local_components"),
        ("component_recorder", True),
        ("timing_boundary", "cuda_events_local_components"),
        ("warmup", 0),
        ("iterations", 2),
    ],
)
def test_forward_admission_rejects_unqualified_timing_contract(attempt, field, value):
    path = attempt / "execution-contract.json"
    contract = json.loads(path.read_text())
    path.write_text(json.dumps(contract | {field: value}))
    with pytest.raises(ValueError, match="timing contract is not qualified"):
        aggregate_forward_results(attempt)


@pytest.mark.parametrize(
    "flag",
    ["--disable-custom-all-reduce", "--enforce-disable-flashinfer-allreduce-fusion", "--disable-shared-experts-fusion"],
)
def test_forward_admission_requires_each_unfused_execution_flag(attempt, flag):
    path = attempt / "execution-contract.json"
    contract = json.loads(path.read_text())
    contract["native_cli_args"].remove(flag)
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="collective/shared-expert contract"):
        aggregate_forward_results(attempt)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("value", [None, "cuda_graph"])
def test_forward_admission_requires_explicit_eager_graph_backends(attempt, phase, value):
    path = attempt / "execution-contract.json"
    contract = json.loads(path.read_text())
    argv = contract["native_cli_args"]
    index = argv.index(f"--cuda-graph-backend-{phase}")
    if value is None:
        del argv[index : index + 2]
    else:
        argv[index + 1] = value
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="eager execution"):
        aggregate_forward_results(attempt)


def test_forward_admission_rejects_decoder_profile_disagreement(attempt):
    path = attempt / "input_provenance.json"
    inputs = json.loads(path.read_text())
    path.write_text(json.dumps(inputs | {"execution_profile": "decoder_bounded"}))
    with pytest.raises(ValueError, match="decoder profile mismatch"):
        aggregate_forward_results(attempt)


def test_forward_admission_rejects_mutated_frozen_workload(attempt):
    path = attempt / "workload-plan.json"
    plan = json.loads(path.read_text())
    plan["cases"][0]["prefix"] += 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan differs from frozen source"):
        aggregate_forward_results(attempt)
