# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY CPU fixtures for independent FPM and Ops serving-policy acceptance."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from collector.fpm_forward import glm53flash_validation as validation

from tests.unit.collector.test_glm53flash_validation import campaign as _campaign_fixture
from tests.unit.collector.test_glm53flash_validation import write_plan

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    return _campaign_fixture.__wrapped__(tmp_path, monkeypatch)


def policy(seed=1):
    return {
        "random_seed": seed,
        "mem_fraction_static": 0.9062999999999999,
        "max_running_requests": 32,
        "cuda_graph_config": {"decode": {"backend": "cuda_graph"}, "prefill": {"backend": "eager"}},
        "cuda_graph_bs_decode": [1, 2, 4, 8, 16, 32],
        "port": 30000,
        "api_key": None,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("mem_fraction_static", 0.82),
        ("max_running_requests", 16),
        ("cuda_graph_config", {"decode": {"backend": "eager"}, "prefill": {"backend": "eager"}}),
        ("cuda_graph_bs_decode", [1, 2, 4, 8, 16]),
        ("port", 30001),
        ("future_native_option", True),
    ],
)
@pytest.mark.parametrize("mode", ["fpm", "ops"])
def test_different_actual_settings_reject_before_prediction_and_keep_all_points(
    campaign, tmp_path, monkeypatch, field, value, mode
):
    campaign["mode"] = mode
    original = validation._native_run
    predict = validation._predict
    predicted_backends = []

    def native(run, base):
        result = original(run, base)
        if run["key"][0] == "sglang":
            config = policy()
            if run["role"] == "holdout":
                config[field] = value
            result.update(validation._sglang_execution_policy(config))
        return result

    def prediction(run, *args, **kwargs):
        predicted_backends.append(run["key"][0])
        return predict(run, *args, **kwargs)

    monkeypatch.setattr(validation, "_native_run", native)
    monkeypatch.setattr(validation, "_predict", prediction)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "FAILED"
    assert report["coverage"]["requested_points"] == 48
    assert report["coverage"]["compared_points"] == 24
    assert predicted_backends == ["vllm"] * 8
    for cell in report["cells"][8:]:
        assert cell["acceptance"] == "FAILED"
        assert "execution policies differ across calibration/holdout" in cell["errors"][0]["error"]
        assert all(row["status"] == "MEASURED_NO_PREDICTION" for row in cell["points"])


@pytest.mark.parametrize("mode", ["fpm", "ops"])
def test_only_random_seed_is_normalized_and_private_settings_are_not_reported(campaign, tmp_path, monkeypatch, mode):
    campaign["mode"] = mode
    original = validation._native_run

    def native(run, base):
        result = original(run, base)
        if run["key"][0] == "sglang":
            config = policy(1 if run["role"] == "calibration" else 9237)
            config["api_key"] = "TEST_ONLY_MUST_NOT_BE_PUBLISHED"
            result.update(validation._sglang_execution_policy(config))
        return result

    monkeypatch.setattr(validation, "_native_run", native)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "PASSED"
    assert "TEST_ONLY_MUST_NOT_BE_PUBLISHED" not in json.dumps(report)
    assert "_execution_policy" not in json.dumps(report)
    for cell in report["cells"][8:]:
        assert cell["calibration_evidence"]["execution_policy"] == cell["holdout_evidence"]["execution_policy"]


@pytest.mark.parametrize("damage", ["missing", "empty", "digest", "normalization"])
@pytest.mark.parametrize("mode", ["fpm", "ops"])
def test_missing_or_inconsistent_policy_is_not_accepted(campaign, tmp_path, monkeypatch, damage, mode):
    campaign["mode"] = mode
    original = validation._native_run

    def native(run, base):
        result = original(run, base)
        if run["key"][0] == "sglang" and run["role"] == "holdout":
            if damage == "missing":
                result.pop("_execution_policy")
            elif damage == "empty":
                result["_execution_policy"] = {}
            else:
                result["execution_policy"]["sha256" if damage == "digest" else "normalization"] = "wrong"
        return result

    monkeypatch.setattr(validation, "_native_run", native)
    report = validation.evaluate(campaign, tmp_path)
    assert report["acceptance"] == "FAILED"
    assert report["coverage"]["requested_points"] == 48
    assert report["coverage"]["compared_points"] == 24


@pytest.mark.parametrize("different", [False, True])
@pytest.mark.parametrize("mode", ["fpm", "ops"])
def test_shard_union_requires_one_policy_and_keeps_private_values_out_of_receipts(
    tmp_path, monkeypatch, different, mode
):
    children = [
        {
            "key": ("sglang", "fp8", 2, "prefill"),
            "role": "holdout",
            "cell": {"cell_id": f"TEST_ONLY-child-{index}"},
            "plan": {"sha256": str(index)},
            "original_point_ids": {1: index + 1},
            "points": [{"benchmark_id": 1}],
        }
        for index in range(2)
    ]
    parent = {
        "children": children,
        "key": children[0]["key"],
        "role": "holdout",
        "points": [{"benchmark_id": i} for i in (1, 2)],
    }

    def native(run, base):
        config = policy(seed=run["original_point_ids"][1])
        config["api_key"] = "TEST_ONLY_SECRET"
        if different and run is children[1]:
            config["mem_fraction_static"] = 0.82
        return {
            "values": {1: 12},
            "request_ids": {run["cell"]["cell_id"]},
            "backend_version": "0.5.20",
            "receipts": [],
            "timing_boundary": "TEST_ONLY_GPU_FORWARD",
            **validation._sglang_execution_policy(config),
        }

    monkeypatch.setattr(validation, "_native_run", native)
    from collector import glm53flash_validation as ops_validation

    monkeypatch.setattr(ops_validation, "load_native", native)
    if different:
        with pytest.raises(ValueError, match="execution policies differ across native shards"):
            validation._load_native(parent, tmp_path, mode)
    else:
        result = validation._load_native(parent, tmp_path, mode)
        assert result["values"] == {1: 12, 2: 12}
        assert "TEST_ONLY_SECRET" not in json.dumps(result["shards"])
        assert "_execution_policy" not in json.dumps(result["shards"])
        assert len({child["execution_policy"]["sha256"] for child in result["shards"]}) == 1


@pytest.mark.parametrize("damage", [None, "rank_policy", "receipt_sha", "missing"])
def test_native_adapter_reads_actual_sha_bound_resolved_config(tmp_path, monkeypatch, damage):
    spec = write_plan(tmp_path, ("sglang", "fp8", 2, "prefill"), "holdout")
    run = validation._plan_run(spec, tmp_path, "holdout")
    root = tmp_path / spec["raw_root"]
    root.mkdir()

    def receipt(name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return {"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for rank in range(2):
        config = policy(seed=rank)
        if damage == "rank_policy" and rank == 1:
            config["mem_fraction_static"] = 0.82
        resolved = receipt(f"TEST_ONLY-resolved-{rank}.json", config)
        if damage == "receipt_sha" and rank == 1:
            resolved["sha256"] = "0" * 64
        manifest = {
            "resolved_config": resolved,
            "requests": receipt(
                f"TEST_ONLY-requests-{rank}.json", {"dataset_role": "holdout", "requests": {"TEST_ONLY-request": {}}}
            ),
        }
        receipt(
            f"benchmark-rank-{rank}.json",
            {
                "artifact_type": "rank",
                "input_provenance": {"native_forward_manifest": manifest} if damage != "missing" else {},
            },
        )
    monkeypatch.setattr(
        validation,
        "validate_native_collection",
        lambda *args, **kwargs: SimpleNamespace(
            points=[SimpleNamespace(point=point, rank_wall_times=((0, 0.012), (1, 0.011))) for point in run["points"]],
            input_provenance={
                "text_sha256": run["corpus"],
                "tokenizer_revision": validation.MODEL_REVISIONS[run["plan"]["model_path"]],
            },
            runtime_run_id="TEST_ONLY-run",
            runtime_grid_digest="TEST_ONLY-grid",
            backend_version="0.5.20",
        ),
    )
    if damage:
        with pytest.raises(ValueError, match="policies differ|digest mismatch|identities are missing"):
            validation._native_run(run, tmp_path)
    else:
        result = validation._native_run(run, tmp_path)
        assert result["execution_policy"] == validation._sglang_execution_policy(policy())["execution_policy"]
        assert result["_execution_policy"] == {k: v for k, v in policy().items() if k != "random_seed"}
        assert len(result["receipts"]) == 6
