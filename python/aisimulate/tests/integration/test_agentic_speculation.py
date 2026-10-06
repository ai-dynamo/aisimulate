# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing SDK schemes reach canonical cost construction and AgentX replay."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tests.unit.sdk.speculation.test_replay_config import PUBLIC, SCHEMES, TARGET

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate_core.sdk import RustForwardPassPerfModel

pytestmark = pytest.mark.integration
WEKA = Path(__file__).resolve().parents[4] / "tests/e2e/configs/unified_cli/fixtures/traces/weka-relative.json"


def _run(tmp_path, cost, mode="aggregated", backend="vllm", seed=73, output=9, auto_capacity=False):
    source = json.loads(WEKA.read_text())
    lengths = []

    def update(requests):
        for request in requests:
            if request["type"] == "s":
                request["out"] = output if lengths else 1
                lengths.append(request["out"])
            elif "requests" in request:
                update(request["requests"])

    update(source["requests"])
    path = tmp_path / "weka.json"
    path.write_text(json.dumps(source))
    worker = {
        "scheduler": {"max_batched_tokens": 512, "max_sequences": 8},
        "kv_cache": {"block_size": 16, "capacity": {"type": "fixed", "blocks": 2048}},
    }
    if auto_capacity:
        worker["kv_cache"].pop("capacity")
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "mode": mode,
                "model": TARGET,
                "backend": backend,
                "hardware": "h200_sxm",
                "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
                "context_length": 4096,
                "speculation": {**deepcopy(cost), "expected_accepted_tokens": 1.5, "seed": seed},
                "workers": {
                    role: deepcopy(worker)
                    for role in (("aggregated",) if mode == "aggregated" else ("prefill", "decode"))
                },
            },
            "traffic": {
                "source": {"type": "trace", "format": "weka", "paths": [str(path)], "block_size": 4},
                "load": {"type": "trace_timestamps", "agentic_lanes": 1},
            },
        }
    )
    saved = CorePredictionConfig.model_validate_json(config.model_dump_json())
    assert saved == config
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(saved), output_requirements=ReplayOutputRequirements(capture_per_request=True)
        )
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == len(lengths) == 4
    assert report.metrics["total_output_tokens"] == sum(lengths)
    assert sorted(row["output_length"] for row in report.metadata["native_report"]["per_request"]) == sorted(lengths)
    return report.metadata["native_report"]["per_request"]


@pytest.mark.parametrize("cost,depth,width", SCHEMES)
def test_canonical_scheme_metadata_and_resolved_identity(cost, depth, width):
    public = PUBLIC.validate_python({**cost, "expected_accepted_tokens": 1.5})
    request = {
        "model": TARGET,
        "backend": "vllm",
        "backend_version": "0.24.0",
        "system": "h200_sxm",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        "speculation": public.cost_config(),
    }
    model = RustForwardPassPerfModel.best_available(request)
    try:
        metadata = model.speculation_metadata()
        assert metadata["kind"] == cost["kind"]
        assert (metadata["max_accepted_draft_tokens"], metadata["verify_width"]) == (depth, width)
        assert (
            metadata["draft_weights_bytes"] > 0
            if cost["kind"] not in {"mtp", "ngram"}
            else metadata["draft_weights_bytes"] == 0
        )
        assert model.static_phase_latency(batch_size=1, input_tokens=128, output_tokens=2, prefill=False) > 0
        assert model.diagnostics()["provenance"]["config"]["speculation"] == public.cost_config()
    finally:
        model.close()


@pytest.mark.parametrize(
    "cost,backend",
    [(cost, "vllm") for cost, _, _ in SCHEMES] + [(SCHEMES[index][0], "sglang") for index in (1, 2, 3, 5)],
)
@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_schemes_complete_weka_dag_and_short_tails(tmp_path, cost, backend, mode):
    _run(tmp_path, cost, mode=mode, backend=backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_tree_seed_reproduces_worker_lifecycle_and_auto_capacity(tmp_path, backend):
    cost = SCHEMES[3][0]
    first = _run(tmp_path, cost, backend=backend, output=128, auto_capacity=True)
    assert _run(tmp_path, cost, backend=backend, output=128, auto_capacity=True) == first
    assert _run(tmp_path, cost, backend=backend, output=128, seed=74, auto_capacity=True) != first


def test_real_eagle_does_not_relabel_unsupported_target():
    request = {
        "model": "MiniMaxAI/MiniMax-M2.7",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "system": "h200_sxm",
        "worker_type": "aggregated",
        "tp": 4,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "speculation": SCHEMES[2][0],
    }
    with pytest.raises(ValueError, match="EAGLE-3 modeling supports model families"):
        RustForwardPassPerfModel.best_available(request)


@pytest.mark.parametrize("cost", [SCHEMES[4][0], SCHEMES[6][0]])
def test_scheme_backend_validation_remains_authoritative(cost):
    request = {
        "model": TARGET,
        "backend": "sglang",
        "backend_version": "0.5.14",
        "system": "h200_sxm",
        "worker_type": "aggregated",
        "speculation": cost,
    }
    with pytest.raises(ValueError, match="modeling supports backends"):
        RustForwardPassPerfModel.best_available(request)
