# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit MTP reuses legacy costs and progress through public Agentic replay."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import MtpSpeculationConfig

pytestmark = [pytest.mark.integration, pytest.mark.pre_merge, pytest.mark.gpu_0]
WEKA = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-relative.json"
GLM, DSV4 = "nvidia/GLM-5.2-NVFP4", "deepseek-ai/DeepSeek-V4-Pro"
MTP = {"kind": "mtp", "num_speculative_tokens": 3, "expected_accepted_tokens": 1.5, "seed": 42}


def _prediction(path=WEKA, model=GLM, backend="sglang", topology="aggregated"):
    worker = {
        "parallelism": {"tensor": 8, "moe_tensor": 8},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
        "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 4096}},
    }
    return {
        "engine": {
            "mode": topology,
            "model": model,
            "backend": backend,
            "hardware": "b300_sxm",
            "backend_version": "0.5.14" if backend == "sglang" else "0.24.0",
            "context_length": 262144,
            "speculation": deepcopy(MTP),
            "workers": {
                role: deepcopy(worker)
                for role in (("aggregated",) if topology == "aggregated" else ("prefill", "decode"))
            },
        },
        "traffic": {
            "source": {"type": "trace", "format": "weka", "paths": [str(path)], "block_size": 4},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }


def _run(raw):
    config = CorePredictionConfig.model_validate(raw)
    saved = CorePredictionConfig.model_validate_json(config.model_dump_json())
    assert saved.engine.speculation == config.engine.speculation
    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(
            prediction_to_replay_spec(saved), output_requirements=ReplayOutputRequirements(capture_per_request=True)
        )
    finally:
        runner.close()


def _weka_with_output(tmp_path, tokens):
    source = json.loads(WEKA.read_text())

    def extend(requests):
        for request in requests:
            if request["type"] == "s":
                request["out"] = tokens
            elif "requests" in request:
                extend(request["requests"])

    extend(source["requests"])
    path = tmp_path / "weka.json"
    path.write_text(json.dumps(source))
    return path


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_speculative_tokens", 0),
        ("num_speculative_tokens", 6),
        ("num_speculative_tokens", True),
        ("expected_accepted_tokens", -1),
        ("expected_accepted_tokens", 3.1),
        ("expected_accepted_tokens", float("nan")),
        ("expected_accepted_tokens", True),
        ("seed", -1),
        ("seed", 1 << 64),
        ("seed", True),
    ],
)
def test_invalid_mtp_assumption(field, value):
    with pytest.raises(ValidationError):
        MtpSpeculationConfig.model_validate({**MTP, field: value})


@pytest.mark.parametrize("expected,rates", [(0, [0, 0, 0]), (1.5, [1, 0.5, 0]), (3, [1, 1, 1])])
def test_mtp_separates_acceptance_from_cost(expected, rates):
    config = MtpSpeculationConfig.model_validate({**MTP, "expected_accepted_tokens": expected})
    assert config.acceptance_rates == rates
    assert config.cost_config() == {"kind": "mtp", "params": {"num_speculative_tokens": 3}}
    with pytest.raises(ValidationError, match="expected_accepted_tokens"):
        MtpSpeculationConfig.model_validate({"kind": "mtp", "num_speculative_tokens": 3})


def test_explicit_method_rejects_legacy_conflicts():
    raw = _prediction()
    raw["engine"].update(nextn=3, nextn_accepted=1.5)
    with pytest.raises(ValidationError, match="cannot be combined"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("model,backend", [(GLM, "vllm"), (GLM, "sglang"), (DSV4, "sglang")])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_weka_override_and_legacy_complete_identically(tmp_path, model, backend, topology):
    raw = _prediction(_weka_with_output(tmp_path, 8), model, backend, topology)
    explicit = _run(raw)
    raw["engine"].pop("speculation")
    raw["engine"].update(nextn=3, nextn_accepted=1.5)
    legacy = _run(raw)
    wall_metrics = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    assert {k: v for k, v in explicit.metrics.items() if k not in wall_metrics} == {
        k: v for k, v in legacy.metrics.items() if k not in wall_metrics
    }
    assert explicit.metrics["completed_requests"] == 4
    assert explicit.metrics["total_output_tokens"] == 32
    assert explicit.metadata["agentic_model_projection"]["target_model"] == model
    assert [row["output_length"] for row in explicit.metadata["native_report"]["per_request"]] == [8] * 4


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_one_token_weka_tails_complete_once(backend, topology):
    report = _run(_prediction(backend=backend, topology=topology))
    assert report.metrics["completed_requests"] == report.metrics["total_output_tokens"] == 4
    assert [row["output_length"] for row in report.metadata["native_report"]["per_request"]] == [1] * 4


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_public_mtp_seed_controls_fractional_progress(tmp_path, backend):
    raw = _prediction(_weka_with_output(tmp_path, 128), backend=backend)
    observations = []
    for seed in (73, 73, 74):
        raw["engine"]["speculation"]["seed"] = seed
        report = _run(raw)
        assert report.metrics["completed_requests"] == 4
        assert report.metrics["total_output_tokens"] == 512
        observations.append(report.metadata["native_report"]["per_request"])
    assert observations[0] == observations[1]
    assert observations[0] != observations[2]
