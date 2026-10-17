# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate CLI owns routed AgentX execution without a Dynamo Python runner."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_host_offload_cli import _prediction_engine, _RecordingRuntime
from test_host_resources import _supervised

from aisimulate import main as cli
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config_adapter import PredictionAdapterContext, RecommendationAdapterContext, resolve_config_adapters
from aisimulate.router import EngineRouterConfigAdapter, RouterConfig
from aisimulate.runner import EngineReplayRunnerFactory, InvalidRunnerError
from aisimulate.sweeper.provider import CandidateContext, SweepContext


def _routed_spec(*, affinity="session", mode="aggregated"):
    public = CorePredictionConfig.model_validate({"engine": _prediction_engine(mode=mode)})
    adapter = EngineRouterConfigAdapter().compile_prediction(
        {"policy": "kv_router", **({"affinity": {"mode": affinity}} if affinity else {})},
        PredictionAdapterContext(engine={}, traffic={}, evaluation={}),
    )
    return prediction_to_replay_spec(public, adapter_specs={"engine.router": adapter})


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("affinity", ["session", "sibling_group"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_agentx_cli_uses_aisimulate_native_runtime_without_dynamo_plugin(
    tmp_path, monkeypatch, capsys, mode, affinity, backend
) -> None:
    # This unit test invokes the child CLI directly to record its runtime calls.
    # Model its supervisor budget; the subprocess tests below use real supervision.
    _supervised(monkeypatch)
    budget = json.loads(os.environ["_AISIMULATE_SUPERVISED_BUDGET"])
    budget.update(events_path=str(tmp_path / "events.jsonl"), ready_path=str(tmp_path / "ready"))
    monkeypatch.setenv("_AISIMULATE_SUPERVISED_BUDGET", json.dumps(budget))
    runtime = _RecordingRuntime()
    imported = []
    original_import = importlib.import_module

    def load(name, *args, **kwargs):
        assert not name.startswith("dynamo"), "routing must not import a Dynamo Python runner"
        if name == "aisimulate._runtime":
            imported.append(name)
            return runtime
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", load)
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kwargs: pytest.fail("built-in routing must not discover external integration entry points"),
    )
    engine = _prediction_engine(mode=mode)
    engine["backend"] = backend
    if mode == "disaggregated":
        engine["kv_transfer"] = {"bytes_per_token": 131_072}
    config = {
        "engine": engine,
        "traffic": {
            "source": {
                "type": "trace",
                "paths": [str(Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-g2-restore.jsonl")],
                "format": "weka",
            },
            "load": {"type": "trace_timestamps", "agentic_lanes": 2, "agentic_snapshot": {"seed": 7}},
        },
        "router": {"policy": "kv_router", "affinity": {"mode": affinity}},
    }
    path = tmp_path / "agentx.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(tmp_path / "out"), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1
    assert imported == ["aisimulate._runtime"]
    execution = runtime.execution_spec
    assert execution["spec"]["adapters"]["placement"] == {
        "provider": "dynamo_kv_router",
        "config": {"router_config": {}, "affinity": {"mode": affinity, "ttl_seconds": 3600}},
    }
    assert execution["traffic"]["agentic_lanes"] == 2
    assert execution["traffic"]["agentic_snapshot"] == {"seed": 7}
    assert execution["traffic"]["trace_format"] == "weka"


def test_builtin_router_resolution_does_not_enumerate_plugins(monkeypatch) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: pytest.fail("unrelated plugin discovery"))
    assert isinstance(resolve_config_adapters(["engine.router"])["engine.router"], EngineRouterConfigAdapter)


@pytest.mark.parametrize("policy", ["round_robin", "kv_router"])
def test_fixed_router_recommendation_preserves_config_and_runtime_hook(policy) -> None:
    adapter = EngineRouterConfigAdapter()
    context = RecommendationAdapterContext(
        engine={}, traffic={}, evaluation={}, optimization={}, sweep=SweepContext({}, {}, {})
    )
    plan = adapter.compile_recommendation({"policy": policy}, context)
    candidate = CandidateContext(sample={}, backend_deployment=_routed_spec().backend_deployment)
    concrete = adapter.materialize_candidate(plan, {}, candidate)
    assert concrete.config["policy"] == policy
    assert concrete.runtime_hooks == plan.potential_runtime_hooks
    with pytest.raises(ValidationError):
        adapter.compile_recommendation({"policy": {"choices": ["round_robin", "kv_router"]}}, context)


@pytest.mark.parametrize(
    "config",
    [
        {"policy": "round_robin", "affinity": {"mode": "session"}},
        {"policy": "round_robin", "temperature": 0},
        {"policy": "kv_router", "temperature": None},
        {"policy": "kv_router", "temperature": True},
        {"policy": "kv_router", "temperature": float("nan")},
        {"policy": "kv_router", "overlap_score_credit": 1.1},
        {"policy": "kv_router", "prefill_load_model": {"type": "ais"}},
        {"policy": "kv_router", "affinity": {"mode": "unknown"}},
        {"policy": "kv_router", "affinity": {"mode": "session", "ttl_seconds": True}},
        {"policy": "kv_router", "affinity": {"mode": "session", "ttl_seconds": 0}},
        {"policy": "kv_router", "affinity": {"mode": "session", "ttl_seconds": 31_536_001}},
        {"policy": "kv_router", "affinity": {"mode": "session"}, "temperature": 0.5},
        {"policy": "kv_router", "router_config": {"custom_classifier": "ignored"}},
    ],
)
def test_invalid_or_unsupported_routing_configuration_fails_explicitly(config) -> None:
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(config)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("host_offload", "device-only KV cache"),
        ("g3_offload", "device-only KV cache"),
        ("speculation", "speculative decoding disabled"),
        ("backend", "backend=vllm or sglang"),
        ("grouped", "grouped KV caches"),
    ],
)
def test_unsupported_affinity_scope_never_calls_runtime(fault, message) -> None:
    spec = _routed_spec()
    deployment = spec.backend_deployment
    rank = dict(deployment.agg_engine_args)
    if fault == "host_offload":
        rank["native_host_offload"] = {"num_host_blocks": 8}
    elif fault == "g3_offload":
        rank["g3_offload"] = {}
    elif fault == "speculation":
        rank["aic_nextn"] = 1
    elif fault == "grouped":
        rank["kv_cache_groups"] = [{"group_id": 0}]
    else:
        deployment = replace(deployment, backend="trtllm")
    spec = replace(spec, backend_deployment=replace(deployment, agg_engine_args=rank))
    runtime = _RecordingRuntime()
    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec is None


def test_feature_absent_fails_without_round_robin_fallback() -> None:
    class WithoutNativeRouter(_RecordingRuntime):
        def native_router_available(self):
            return False

    runtime = WithoutNativeRouter()
    with pytest.raises(InvalidRunnerError, match="native routing support"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(_routed_spec())
    assert runtime.execution_spec is None


def test_ordinary_kv_router_numeric_parameters_reach_native_policy() -> None:
    spec = _routed_spec(affinity=None)
    adapter = EngineRouterConfigAdapter().compile_prediction(
        {"policy": "kv_router", "overlap_score_credit": 0.5, "prefill_load_scale": 2, "temperature": 0.2},
        PredictionAdapterContext(engine={}, traffic={}, evaluation={}),
    )
    runtime = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(replace(spec, adapters={"engine.router": adapter}))
    assert runtime.execution_spec["spec"]["adapters"]["placement"]["config"]["router_config"] == {
        "overlap_score_credit": 0.5,
        "prefill_load_scale": 2,
        "router_temperature": 0.2,
    }


def test_explicit_round_robin_preserves_existing_native_placement() -> None:
    spec = _routed_spec()
    adapter = EngineRouterConfigAdapter().compile_prediction(
        {"policy": "round_robin"}, PredictionAdapterContext(engine={}, traffic={}, evaluation={})
    )
    runtime = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(replace(spec, adapters={"engine.router": adapter}))
    assert runtime.execution_spec["spec"]["adapters"]["placement"] == {"provider": "round_robin", "config": None}


def test_nested_rank_model_identity_reaches_agentx_driver() -> None:
    spec = _routed_spec()
    rank = {
        "backend": "vllm",
        "num_gpu_blocks": 128,
        "block_size": 4,
        "timing_model": {"type": "external", "provider": "aic", "config": {"model": " nested-target "}},
    }
    spec = replace(
        spec,
        backend_deployment=replace(
            spec.backend_deployment, agg_engine_args={"rank": rank}, performance_model_metadata={}
        ),
        workload={"source_type": "trace", "trace_format": "weka", "load_type": "trace_timestamps", "agentic_lanes": 1},
    )
    runtime = _RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec["traffic"]["execution_model"] == "nested-target"


@pytest.mark.parametrize(("backend", "mode"), [("vllm", "aggregated"), ("sglang", "disaggregated")])
@pytest.mark.parametrize(
    ("affinity", "tokenization"),
    [
        ("session", None),
        ("sibling_group", None),
        ("session", {"workers": 1, "base_ms": 3.0, "per_input_token_us": 500.0}),
    ],
)
@pytest.mark.timeout(60)
def test_native_agentx_routing_through_supervised_cli(tmp_path, backend, mode, affinity, tokenization) -> None:
    runtime = pytest.importorskip("aisimulate._runtime")
    if not getattr(runtime, "native_router_available", lambda: False)():
        pytest.skip("requires the AISimulate native routing wheel")
    fixtures = Path(__file__).parent / "e2e/configs/unified_cli/fixtures"
    engine = _prediction_engine(mode=mode)
    engine.update(backend=backend, model=str(fixtures / "tiny-model"))
    if tokenization is not None:
        engine["tokenization"] = tokenization
    for worker in engine["workers"].values():
        worker["kv_cache"]["block_size"] = 4
        worker["parallelism"].update(tensor=1, replicas=2)
    if mode == "disaggregated":
        engine["kv_transfer"] = {"bytes_per_token": 131_072}
    config = {
        "engine": engine,
        "traffic": {
            "source": {"type": "trace", "paths": [str(fixtures / "traces/weka-g2-restore.jsonl")], "format": "weka"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
        "router": {"policy": "kv_router", "affinity": {"mode": affinity}},
    }
    path = tmp_path / "native-agentx.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "-c",
            str(path),
            "--output-dir",
            str(output),
            "--capture-per-request",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["completed_requests"] == 4
    assert json.loads((output / "resource-runtime.json").read_text())["status"] == "completed"
    events = [json.loads(line) for line in (output / "execution-events.jsonl").read_text().splitlines()]
    plan = next(event["value"] for event in events if event["event"] == "resource_plan")
    assert plan["estimate"]["allocation_model"] == "dynamo-unqualified-v1"
    assert plan["estimate"]["estimated_peak_bytes"] is None
    assert plan["effective_parallelism"] == 1
    report = json.loads((output / "prediction.json").read_text())
    assert len(report["per_request"]) == 4
    for record in report["per_request"]:
        if tokenization is None:
            assert "tokenized_ms" not in record
        else:
            service_ms = tokenization["base_ms"] + tokenization["per_input_token_us"] / 1000 * record["input_length"]
            assert record["tokenized_ms"] - record["arrival_time_ms"] >= service_ms
            assert record["first_admit_ms"] >= record["tokenized_ms"]
    roles = report["routing_policy"]["roles"]
    assert set(roles) == ({"aggregated"} if mode == "aggregated" else {"prefill", "decode"})
    assert roles["aggregated" if mode == "aggregated" else "prefill"]["physical_kv_events"] > 0
    if mode == "disaggregated":
        # Dynamo's P/D decode policy uses load and affinity; cache feedback is prompt-side.
        assert roles["decode"]["physical_kv_events"] == 0
    for evidence in roles.values():
        assert evidence["native_policy"] == "dynamo.DefaultWorkerSelector"
        assert evidence["decision_count"] == 4
        assert evidence["post_dispatch_checks"] == 4
        targets = {}
        for decision in evidence["decisions"]:
            target = (decision["worker_id"], decision["dp_rank"])
            assert targets.setdefault(decision["group_key"], target) == target
        assert any(decision["binding_reused"] for decision in evidence["decisions"])
    assert report["agentic_qualification"] == "functional_only"
