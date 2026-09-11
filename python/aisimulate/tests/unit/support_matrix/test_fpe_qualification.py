# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json

import pytest
from tools.support_matrix.qualify_fpe_support_matrix import qualify

pytestmark = pytest.mark.unit
SHA = "a" * 40
WHEEL_SHA = "b" * 64
PROBE = {
    "model": "test/dense",
    "system": "b200_sxm",
    "backend": "vllm",
    "backend_version": "1.0",
    "forward_model": "op_level",
}


@pytest.fixture
def payload():
    row = {
        **PROBE,
        "tp_size": 1,
        "pp_size": 1,
        "attention_dp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "cp_size": 1,
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "nextn": 0,
        "attention_backend": None,
        "roles": "agg",
        "status": "PASS",
        "latency_ms": 1.0,
        "source_sha": SHA,
        "source_version": "0.12.0",
    }
    return {
        "metadata": {
            "schema_version": 1,
            "source_sha": SHA,
            "source_version": "0.12.0",
            "wheel_sha256": WHEEL_SHA,
            "workload": {"isl": 256},
            "plan_count": 1,
        },
        "results": [{**row, "phase": p} for p in ("prefill", "decode_start", "decode_end", "mixed")],
    }


def check(tmp_path, payload, **kwargs):
    path = tmp_path / "shard" / "fpe_support_matrix.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload))
    return qualify(
        tmp_path,
        expected_shards=[{"system": "b200_sxm", "backend": "vllm"}],
        expected_sha=SHA,
        expected_wheel_sha256=WHEEL_SHA,
        required_probes=[PROBE],
        **kwargs,
    )


def test_complete_native_reports_qualify(tmp_path, payload):
    assert check(tmp_path, payload)["status_counts"] == {"PASS": 4}


@pytest.mark.parametrize(
    "field,value", [("source_sha", "c" * 40), ("wheel_sha256", "d" * 64), ("schema_version", 2), ("plan_count", 2)]
)
def test_wrong_artifact_identity_or_plan_count_fails(tmp_path, payload, field, value):
    payload["metadata"][field] = value
    with pytest.raises(ValueError):
        check(tmp_path, payload)


@pytest.mark.parametrize("status", ["PERF_DATA_MISSING", "SDK_UNREPRESENTABLE", "QUERY_FAILED", "BUILD_FAILED", "SKIP"])
def test_required_case_cannot_be_skipped_or_failed(tmp_path, payload, status):
    for row in payload["results"]:
        row["status"] = status
    with pytest.raises(ValueError):
        check(tmp_path, payload)


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1, 0, True, None])
def test_invalid_native_values_fail(tmp_path, payload, latency):
    payload["results"][0]["latency_ms"] = latency
    with pytest.raises(ValueError, match="latency"):
        check(tmp_path, payload)


def test_duplicate_and_missing_phases_fail(tmp_path, payload):
    duplicate = copy.deepcopy(payload)
    duplicate["results"].append(duplicate["results"][0])
    with pytest.raises(ValueError, match="duplicate"):
        check(tmp_path, duplicate)
    payload["results"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        check(tmp_path, payload)


def test_expected_shards_cannot_disappear(tmp_path, payload):
    check(tmp_path, payload)
    with pytest.raises(ValueError, match="missing shards"):
        qualify(
            tmp_path,
            expected_shards=[{"system": "b200_sxm", "backend": b} for b in ("vllm", "sglang")],
            expected_sha=SHA,
            expected_wheel_sha256=WHEEL_SHA,
            required_probes=[PROBE],
        )


def test_duplicate_shard_rejected(tmp_path, payload):
    check(tmp_path, payload)
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "fpe_support_matrix.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        check(tmp_path, payload)


def test_row_sha_is_checked_independently(tmp_path, payload):
    payload["results"][0]["source_sha"] = "c" * 40
    with pytest.raises(ValueError, match="row source"):
        check(tmp_path, payload)


def test_legitimate_prefill_only_topology_is_complete(tmp_path, payload):
    row = {**payload["results"][0], "tp_size": 2, "roles": "prefill"}
    payload["results"].append(row)
    payload["metadata"]["plan_count"] = 2
    assert check(tmp_path, payload)["status_counts"] == {"PASS": 5}


def test_exploratory_unsupported_case_is_visible_without_qualifying_it(tmp_path, payload):
    extra = [
        {**r, "model": "test/unsupported", "status": "MODEL_UNSUPPORTED", "latency_ms": None}
        for r in payload["results"]
    ]
    payload["results"].extend(extra)
    payload["metadata"]["plan_count"] = 2
    assert check(tmp_path, payload)["status_counts"] == {"PASS": 4, "MODEL_UNSUPPORTED": 4}
