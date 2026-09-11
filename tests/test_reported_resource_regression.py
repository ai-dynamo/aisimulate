# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic preflight regression; native CLI evidence has a separate runner."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import aisimulate.main as cli
import aisimulate.resources as resources
from aisimulate.config import CoreRecommendationConfig
from aisimulate.config.common import split_config_sections
from aisimulate.resources import GIB, HostResources, build_plan, workload_bounds

CASE = Path(__file__).parent / "e2e/configs/resource_safety/reported-mac-sweep.yaml"


def normalized_digest(raw):
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_original_sweep_identity_and_largest_candidate():
    raw = yaml.safe_load(CASE.read_text())
    original = deepcopy(raw)
    # Locks every behavioral field, including all seven concurrency choices.
    assert (
        normalized_digest(raw)
        == "af1a9d53df52c5eab2c07f9be20b69457e51ddd4c43933c42bcca7e489698577"
    )
    core, _ = split_config_sections(raw, command="recommend")
    bounds = workload_bounds(CoreRecommendationConfig.model_validate(core))
    assert bounds["concurrency"] == 64_512
    assert bounds["request_count"] == 6_451_200
    plan = build_plan(
        bounds,
        stack="dynamo",
        host=HostResources(32 * GIB, 16 * GIB, 8),
        requested_parallelism=8,
    )
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["input_token_bytes"] == 264_241_152_000
    assert plan["effective_parallelism"] == 0
    assert raw == original


@pytest.mark.parametrize("largest_only", [False, True])
def test_sweep_refuses_before_candidate_sampling_or_native_allocation(
    tmp_path, monkeypatch, largest_only
):
    raw = yaml.safe_load(CASE.read_text())
    if largest_only:
        raw["traffic"]["load"]["concurrency"] = 64_512
    original = deepcopy(raw)
    config = tmp_path / "case.yaml"
    config.write_text(yaml.safe_dump(raw))

    class AllocationSentinel:
        def capabilities(self):
            pytest.fail("must refuse before preparing the candidate search")

        def create(self, worker_id):
            pytest.fail("must refuse before allocating native requests")

    monkeypatch.setattr(
        cli, "resolve_runner_factory", lambda stack: AllocationSentinel()
    )
    monkeypatch.setattr(
        resources, "discover_host", lambda: HostResources(32 * GIB, 16 * GIB, 8)
    )
    output = tmp_path / "result"
    assert (
        cli._main(
            [
                "recommend",
                "--stack",
                "dynamo",
                "--config",
                str(config),
                "--output-dir",
                str(output),
            ]
        )
        == 3
    )
    plan = json.loads((output / "resource-plan.json").read_text())
    assert plan["status"] == "resource_limited"
    assert plan["estimate"]["request_count"] == 6_451_200
    assert plan["estimate"]["allocation_model"] == "dynamo-eager-u32-v1"
    assert yaml.safe_load(config.read_text()) == original


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing-runtime",
        "timeout",
        "wrong-count",
        "old-binding",
        "rss-overshoot",
        "not-reaped",
    ],
)
def test_native_evidence_gate_cannot_mistake_failure_for_safe_refusal(
    tmp_path, failure
):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts/verify_reported_resource_safety.py"
    spec = importlib.util.spec_from_file_location("resource_regression_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = {
        "status": "resource_limited",
        "exit_code": 3,
        "termination_complete": True,
        "peak_observed_rss_bytes": 100,
        "budget": {"memory_limit_bytes": 200},
    }
    plan = {
        "status": "resource_limited",
        "estimate": {
            "request_count": 6_451_200,
            "allocation_model": "dynamo-generated-u32-v1",
        },
    }
    if failure == "missing-runtime":
        runtime.update(status="failed", exit_code=2)
    elif failure == "timeout":
        runtime.update(status="timed_out", exit_code=-15)
    elif failure == "wrong-count":
        plan["estimate"]["request_count"] = 8
    elif failure == "old-binding":
        plan["estimate"]["allocation_model"] = "dynamo-eager-u32-v1"
    elif failure == "rss-overshoot":
        runtime["peak_observed_rss_bytes"] = 201
    elif failure == "not-reaped":
        runtime["termination_complete"] = False
    (tmp_path / "resource-plan.json").write_text(json.dumps(plan))
    if failure:
        with pytest.raises(RuntimeError):
            module.check_refusal(tmp_path, runtime)
    else:
        assert module.check_refusal(tmp_path, runtime) == plan


def test_evidence_hashing_is_bounded_and_source_identity_is_checked(tmp_path):
    import importlib.util
    import io

    path = Path(__file__).parents[1] / "scripts/verify_reported_resource_safety.py"
    spec = importlib.util.spec_from_file_location("resource_regression_hashing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64 * 1024
            return super().read(size)

    payload = b"report evidence" * 10_000
    assert (
        module.stream_digest(BoundedReader(payload))
        == hashlib.sha256(payload).hexdigest()
    )
    loaded, expected = tmp_path / "loaded", tmp_path / "expected"
    loaded.mkdir()
    expected.mkdir()
    (loaded / "main.py").write_text("version = 1")
    (expected / "main.py").write_text("version = 2")
    with pytest.raises(RuntimeError, match="differs"):
        module.source_identity(loaded, expected)
    (loaded / "main.py").write_text("version = 2")
    assert module.source_identity(loaded, expected)["python_files"] == 1
