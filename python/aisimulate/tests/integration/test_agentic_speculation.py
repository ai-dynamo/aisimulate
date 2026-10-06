# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic SDK costs and accepted paths survive canonical construction and AgentX replay."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.unit.sdk.speculation.test_dense_draft_schemes import REPLAY_SCHEMES, REPLAY_TARGET

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate_core.sdk import RustForwardPassPerfModel, models
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.speculation import SpeculationConfig

pytestmark = pytest.mark.integration
WEKA = Path(__file__).resolve().parents[4] / "tests/e2e/configs/unified_cli/fixtures/traces/weka-relative.json"


def _cost_request(cost, model=REPLAY_TARGET, backend="vllm", tp=1):
    return {
        "model": model,
        "system": "b200_sxm",
        "backend": backend,
        "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
        "worker_type": "aggregated",
        "tp": tp,
        "moe_tp_size": tp,
        "moe_ep_size": 1,
        "estimation_mode": "op_level",
        "speculation": deepcopy(cost),
    }


def _prediction(tmp_path, cost, *, model=REPLAY_TARGET, backend="vllm", topology="aggregated", tp=1):
    source = json.loads(WEKA.read_text())
    lengths = []

    def update(requests):
        for request in requests:
            if request["type"] == "s":
                request["out"] = 128 if lengths else 1
                lengths.append(request["out"])
            elif "requests" in request:
                update(request["requests"])

    update(source["requests"])
    path = tmp_path / "weka.json"
    path.write_text(json.dumps(source))
    worker = {
        "parallelism": {"tensor": tp, "moe_tensor": tp},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
        "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 4096}},
    }
    return {
        "engine": {
            "mode": topology,
            "model": model,
            "backend": backend,
            "hardware": "b200_sxm",
            "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
            "context_length": 262144,
            "speculation": {**deepcopy(cost), "expected_accepted_tokens": 1.5, "seed": 42},
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
    assert saved == config
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            prediction_to_replay_spec(saved), output_requirements=ReplayOutputRequirements(capture_per_request=True)
        )
    finally:
        runner.close()
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_output_tokens"] == 385
    assert report.metadata["agentic_model_projection"]["target_model"] == raw["engine"]["model"]
    rows = report.metadata["native_report"]["per_request"]
    assert sorted(row["output_length"] for row in rows) == [1, 128, 128, 128]
    return rows


@pytest.mark.parametrize("cost", [cost for cost, _, _ in REPLAY_SCHEMES])
def test_canonical_scheme_preserves_cost_and_resolved_identity(cost):
    model = RustForwardPassPerfModel.best_available(_cost_request(cost))
    try:
        assert model.static_phase_latency(batch_size=1, input_tokens=128, output_tokens=2, prefill=False) > 0
        saved = model.diagnostics()["provenance"]["config"]["speculation"]
        assert saved["kind"] == cost["kind"] and saved["params"] == cost["params"]
        if cost.get("draft_config"):
            assert saved["draft_config"] == cost["draft_config"]
    finally:
        model.close()


@pytest.mark.parametrize(
    "target,backend",
    [("nvidia/GLM-5.2-NVFP4", "vllm"), ("nvidia/GLM-5.2-NVFP4", "sglang"), ("deepseek-ai/DeepSeek-V4-Pro", "sglang")],
)
def test_mtp_retains_target_graph_cost_and_legacy_weka_progress(tmp_path, target, backend):
    cost = REPLAY_SCHEMES[0][0]
    shape = {"tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1}
    legacy_graph = models.get_model(target, ModelConfig(**shape, nextn=3), backend)
    graph = models.get_model(target, ModelConfig(**shape, speculation=SpeculationConfig(**cost)), backend)
    assert graph.model_path == target and type(graph) is type(legacy_graph)
    assert [op._spec_json() for op in graph.generation_ops] == [op._spec_json() for op in legacy_graph.generation_ops]
    request = _cost_request(cost, target, backend, tp=8)
    explicit = RustForwardPassPerfModel.best_available(request)
    request.pop("speculation")
    legacy = RustForwardPassPerfModel.best_available({**request, "nextn": 3})
    try:
        query = dict(batch_size=1, input_tokens=128, output_tokens=2, prefill=False)
        assert explicit.static_phase_latency(**query) == legacy.static_phase_latency(**query) > 0
    finally:
        explicit.close()
        legacy.close()
    for topology in ("aggregated", "disaggregated"):
        raw = _prediction(tmp_path, cost, model=target, backend=backend, topology=topology, tp=8)
        rows = _run(raw)
        raw["engine"].pop("speculation")
        raw["engine"].update(nextn=3, nextn_accepted=1.5)
        assert _run(raw) == rows


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_eagle_tree_replays_dag_short_tail_and_seeded_lifecycle(tmp_path, backend, topology):
    raw = _prediction(tmp_path, REPLAY_SCHEMES[3][0], backend=backend, topology=topology)
    first = _run(raw)
    assert _run(raw) == first
    raw["engine"]["speculation"]["seed"] = 43
    assert _run(raw) != first
    raw["engine"].update(nextn=3, nextn_accepted=1.5)
    with pytest.raises(ValidationError, match="cannot be combined"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "cost,target,backend,tp,error",
    [
        (REPLAY_SCHEMES[2][0], "MiniMaxAI/MiniMax-M2.7", "vllm", 4, "model families"),
        (REPLAY_SCHEMES[4][0], REPLAY_TARGET, "sglang", 1, "modeling supports backends"),
        (REPLAY_SCHEMES[6][0], REPLAY_TARGET, "sglang", 1, "modeling supports backends"),
        (REPLAY_SCHEMES[0][0], "moonshotai/Kimi-K3", "sglang", 8, "DSPARK"),
    ],
)
def test_existing_scheme_family_and_backend_validation(cost, target, backend, tp, error):
    with pytest.raises(ValueError, match=error):
        RustForwardPassPerfModel.best_available(_cost_request(cost, target, backend, tp=tp))
