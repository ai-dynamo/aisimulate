# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_spec = importlib.util.spec_from_file_location(
    "moe_routing_accuracy", Path(__file__).resolve().parents[3] / "tools" / "moe_routing_accuracy.py"
)
accuracy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(accuracy)


def test_gate_reports_all_missing_models_and_never_claims_accuracy():
    report = accuracy.summarize([], [])
    assert report["status"] == "NOT_READY"
    assert set(report["models"]) == set(accuracy.REQUIRED_MODELS)
    assert all(value["status"] == "INSUFFICIENT_EVIDENCE" for value in report["models"].values())


def test_gate_compares_total_mape_per_model_not_only_pooled_average():
    # Synthetic VALUES here only test the reducer; they are NOT accuracy evidence.
    rows = [
        dict(
            model_id=name,
            actual=dict.fromkeys(accuracy.COMPONENTS, 1),
            **{
                "power-law": dict.fromkeys(accuracy.COMPONENTS, 1.2),
                "measured": dict.fromkeys(accuracy.COMPONENTS, 1.1),
            },
        )
        for name in accuracy.REQUIRED_MODELS
    ]
    assert accuracy.summarize(rows, [])["status"] == "PASS"
    rows[0]["measured"]["total"] = 1.3
    assert accuracy.summarize(rows, [])["status"] == "NOT_READY"
    rows[0]["measured"]["total"] = 1.1
    assert accuracy.summarize(rows, [{"sample": 3}])["status"] == "NOT_READY"


def test_evidence_contract_rejects_leakage_synthetic_and_prefill_as_decode(tmp_path):
    # This is a schema fixture, never submitted to the accuracy report.
    raw = json.dumps([[0, 1], [1, 2]]).encode()
    (tmp_path / "routes.json").write_bytes(raw)
    record = dict(
        phase="prefill",
        route_file="routes.json",
        route_sha256=hashlib.sha256(raw).hexdigest(),
        workload_sha256="a" * 64,
        bundle_workload_sha256s=["b" * 64],
        per_rank_tokens=1,
        model_config={"moe_ep_size": 2},
        top_k=2,
        num_experts=4,
        measurement=dict(
            kind="routing_replay",
            kernel_revision="pinned-test",
            gpu_type="test",
            driver="test",
            timing_method="test",
            placement="contiguous_expert_id",
            latency_ms={"dispatch": 1, "combine": 1, "compute": 2},
        ),
    )
    assert accuracy.validate_observation(record, tmp_path)["total"] == 4
    with pytest.raises(ValueError, match="disjoint"):
        accuracy.validate_observation(record | {"workload_sha256": "b" * 64}, tmp_path)
    with pytest.raises(ValueError, match="decode accuracy"):
        accuracy.validate_observation(record | {"phase": "decode"}, tmp_path)
    record["measurement"]["kind"] = "sampled_from_measured"
    with pytest.raises(ValueError, match="not synthetic"):
        accuracy.validate_observation(record, tmp_path)
