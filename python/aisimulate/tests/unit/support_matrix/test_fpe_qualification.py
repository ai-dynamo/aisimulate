# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools.support_matrix.qualify_fpe_support_matrix import qualify
from tools.verify_installed_package_layers import _exercise_fpe_matrix, _verify_fpe_probe_results

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
    "field,value,message",
    [
        ("source_sha", "c" * 40, "wrong source/schema identity"),
        ("wheel_sha256", "d" * 64, "wrong wheel identity"),
        ("schema_version", 2, "wrong source/schema identity"),
        ("plan_count", 2, "plan count does not match"),
    ],
)
def test_wrong_artifact_identity_or_plan_count_fails(tmp_path, payload, field, value, message):
    payload["metadata"][field] = value
    with pytest.raises(ValueError, match=message):
        check(tmp_path, payload)


@pytest.mark.parametrize(
    "status,message",
    [
        ("PERF_DATA_MISSING", "required known-good probes no longer pass"),
        ("SDK_UNREPRESENTABLE", "required known-good probes no longer pass"),
        ("QUERY_FAILED", "unexpected native probe failure"),
        ("BUILD_FAILED", "unexpected native probe failure"),
        ("SKIP", "invalid probe result"),
    ],
)
def test_required_case_cannot_be_skipped_or_failed(tmp_path, payload, status, message):
    for row in payload["results"]:
        row["status"] = status
    with pytest.raises(ValueError, match=message):
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


@pytest.fixture(params=[False, True], ids=["merged-roles", "distinct-roles"])
def installed_payload(payload, request):
    for row in payload["results"]:
        row["reproducer"] = json.dumps({"compile": {"model_path": "test/dense", "tp_size": 1}})
        row["roles"] = "agg|prefill|decode" if not request.param else "agg"
    if request.param:
        for phase, role, tp_size in (
            ("prefill", "prefill", 2),
            ("decode_start", "decode", 4),
            ("decode_end", "decode", 4),
        ):
            payload["results"].append(
                {
                    **payload["results"][0],
                    "phase": phase,
                    "roles": role,
                    "reproducer": json.dumps({"compile": {"model_path": "test/dense", "tp_size": tp_size}}),
                }
            )
        payload["metadata"]["plan_count"] = 3
    return payload


def test_installed_fpe_accepts_complete_merged_or_distinct_roles(installed_payload):
    _verify_fpe_probe_results(installed_payload)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("status", "QUERY_FAILED", "did not execute"),
        ("latency_ms", float("nan"), "invalid latencies"),
        ("latency_ms", True, "invalid latencies"),
        ("roles", "unknown", "invalid roles"),
    ],
)
def test_installed_fpe_rejects_invalid_probe(installed_payload, field, value, message):
    installed_payload["results"][-1][field] = value
    with pytest.raises(RuntimeError, match=message):
        _verify_fpe_probe_results(installed_payload)


def test_installed_fpe_rejects_incomplete_role(installed_payload):
    installed_payload["results"].pop()
    with pytest.raises(RuntimeError, match="incomplete topology phases"):
        _verify_fpe_probe_results(installed_payload)


def test_installed_fpe_rejects_duplicate_probe(installed_payload):
    installed_payload["results"].append(installed_payload["results"][-1])
    with pytest.raises(RuntimeError, match="duplicate phases"):
        _verify_fpe_probe_results(installed_payload)


def test_installed_fpe_rejects_wrong_plan_count(installed_payload):
    installed_payload["metadata"]["plan_count"] += 1
    with pytest.raises(RuntimeError, match="plan count"):
        _verify_fpe_probe_results(installed_payload)


def test_source_provenance_uses_checkout_instead_of_dispatch_event(monkeypatch):
    from tools.support_matrix.generate_fpe_support_matrix import _source_sha

    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    root = Path(__file__).resolve().parents[5]
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert _source_sha() == expected


def test_installed_native_fpe_ignores_source_package_on_pythonpath(tmp_path, monkeypatch):
    shadow = tmp_path / "shadow"
    package = shadow / "aisimulate"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("source package shadowed the installed wheel")\n')
    source = Path(__file__).resolve().parents[3] / "src"
    monkeypatch.setenv("PYTHONPATH", f"{shadow}:{source}")
    _exercise_fpe_matrix()
