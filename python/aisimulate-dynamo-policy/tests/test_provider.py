# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python integration contract; actual native routing is covered by CLI acceptance."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aisimulate_dynamo_policy import runner as plugin
from aisimulate_dynamo_policy.provider import PROVIDER, DynamoPolicyConfigAdapter

from aisimulate.config_adapter import PredictionAdapterContext, validate_config_adapter
from aisimulate.runner import InvalidRunnerError
from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplayOutputRequirements, ReplaySpec


def _adapter(config=None):
    return DynamoPolicyConfigAdapter().compile_prediction(
        config or {"policy": "kv_router", "affinity": {"mode": "sibling_group", "ttl_seconds": 12}},
        PredictionAdapterContext(engine={}, traffic={}, evaluation={}),
    )


def _spec(adapter=None):
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args={
                "worker_type": "aggregated",
                "engine_type": "vllm",
                "aic_backend": "vllm",
                "aic_model_path": "test-model",
                "aic_system": "test-system",
                "aic_tp_size": 1,
                "aic_attention_dp_size": 1,
                "block_size": 4,
                "num_gpu_blocks": 16,
                "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
            },
            num_workers=2,
        ),
        workload={"isl": 8, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
        goal={"target": "throughput"},
        adapters={PROVIDER: adapter or _adapter()},
    )


def test_provider_public_policy_and_affinity_are_separate():
    validate_config_adapter(DynamoPolicyConfigAdapter(), requested_name=PROVIDER)
    plain = _adapter({"policy": "kv_router"})
    affinity = _adapter()
    assert plain.config == {"policy": "kv_router"}
    assert affinity.config["policy"] == plain.config["policy"]
    assert affinity.runtime_hooks[0].config["affinity"] == {"mode": "sibling_group", "ttl_seconds": 12.0}


@pytest.mark.parametrize("ttl", [None, 1, 1.125, 3600, 31_536_000])
def test_ttl_defaults_and_fractional_seconds_reach_native_json(ttl):
    affinity = {"mode": "session"}
    if ttl is not None:
        affinity["ttl_seconds"] = ttl
    config = _adapter({"policy": "kv_router", "affinity": affinity}).config
    assert json.loads(json.dumps(config))["affinity"]["ttl_seconds"] == (3600 if ttl is None else ttl)


@pytest.mark.parametrize(
    "config",
    [
        {"policy": "round_robin"},
        {"policy": "kv_router", "temperature": 1},
        {"policy": "kv_router", "affinity": {"mode": "unknown"}},
        *[
            {"policy": "kv_router", "affinity": {"mode": "session", "ttl_seconds": ttl}}
            for ttl in (0, 0.5, -1, True, "3600", float("inf"), float("nan"), 31_536_001)
        ],
    ],
)
def test_unsupported_routing_configuration_fails(config):
    with pytest.raises(ValueError):
        _adapter(config)


def test_recommendation_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="offline predict only"):
        DynamoPolicyConfigAdapter().compile_recommendation({}, None)


@pytest.fixture
def native_contracts(monkeypatch):
    core_contract = {"api_version": 1, "core_version": "0.13.0", "core_source_sha256": "a" * 64}
    plugin_contract = {**core_contract, "plugin_version": "0.13.0", "dynamo_revision": plugin.DYNAMO_REVISION}
    native = SimpleNamespace(native_contract=lambda: json.dumps(plugin_contract), run_replay_json=lambda *_: "{}")
    core = SimpleNamespace(native_replay_contract=lambda: core_contract)
    versions = {"aisimulate": "0.13.0", "aisimulate-dynamo-policy": "0.13.0"}
    monkeypatch.setattr(
        plugin.importlib, "import_module", lambda name: core if name == "aisimulate._runtime" else native
    )
    monkeypatch.setattr(plugin.importlib.metadata, "version", versions.__getitem__)
    return core_contract, plugin_contract, versions, native


def test_matching_native_and_python_contracts(native_contracts):
    assert plugin._load_native() is native_contracts[3]
    factory = plugin.DynamoPolicyRunnerFactory()
    assert factory.capabilities().supports_backend_topology("sglang", "disagg")
    assert not factory.capabilities().supports_execution_mode("online")
    assert not factory.capabilities().supports_backend_topology("trtllm", "agg")
    assert factory.capabilities().supports_agentic_snapshots
    assert factory.capabilities().supports_agentic_warmup


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    [
        (0, "api_version", True, "API version"),
        (1, "api_version", 2, "API version"),
        (0, "core_source_sha256", "b" * 64, "different AISimulate core sources"),
        (1, "core_source_sha256", None, "different AISimulate core sources"),
        (1, "core_version", "0.12.0", "mismatched"),
        (1, "plugin_version", "0.12.0", "mismatched"),
        (2, "aisimulate", "0.12.0", "mismatched"),
        (2, "aisimulate-dynamo-policy", "0.14.0", "mismatched"),
        (1, "dynamo_revision", "unknown", "Dynamo revision"),
    ],
)
def test_mixed_builds_fail_closed(native_contracts, target, key, value, message):
    native_contracts[target][key] = value
    with pytest.raises(InvalidRunnerError, match=message):
        plugin._load_native()


def test_missing_native_has_paired_install_message(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(plugin.importlib, "import_module", missing)
    with pytest.raises(InvalidRunnerError, match="matching aisimulate and aisimulate-dynamo-policy wheels"):
        plugin._load_native()


@pytest.mark.parametrize("value", [None, "not-json", "[]", {"core_version": "invalid"}])
def test_invalid_native_contract_has_paired_install_message(native_contracts, value):
    native_contracts[3].native_contract = lambda: value
    with pytest.raises(InvalidRunnerError, match="matching aisimulate and aisimulate-dynamo-policy wheels"):
        plugin._load_native()


def test_missing_native_executor_is_rejected(native_contracts):
    del native_contracts[3].run_replay_json
    with pytest.raises(InvalidRunnerError, match="missing run_replay_json"):
        plugin._load_native()


def test_runner_uses_canonical_materializer_and_native_policy_seam():
    calls = []

    def run_replay_json(payload, router):
        calls.append((json.loads(payload), json.loads(router)))
        return json.dumps(
            {"completed_requests": 1, "mean_ttft_ms": 2.0, "dynamo_policy": {"native_policy": "dynamo.SelectionCore"}}
        )

    native = SimpleNamespace(run_replay_json=run_replay_json)
    replay = plugin.DynamoPolicyReplayRunner(0, 4, native)
    report = replay.run(_spec(), output_requirements=ReplayOutputRequirements(include_raw_report=True))
    assert len(calls) == 1
    payload, config = calls[0]
    assert len(payload["requests"]) == 1
    assert payload["requests"][0]["input_tokens"] == 8
    assert config == _adapter().config
    assert report.metadata["native_report"]["dynamo_policy"]["native_policy"] == "dynamo.SelectionCore"
    assert report.metrics["completed_requests"] == 1


def test_runner_does_not_drop_incompatible_or_unconfigured_hooks():
    native = SimpleNamespace(run_replay_json=lambda *_: pytest.fail("invalid hook must not execute"))
    replay = plugin.DynamoPolicyReplayRunner(0, 4, native)
    with pytest.raises(ValueError, match="exactly one router"):
        replay.run(replace(_spec(), adapters={}))
    adapter = _adapter()
    bad = AdapterReplaySpec(
        config=adapter.config,
        runtime_hooks=(RuntimeHookSpec(PROVIDER, "placement_policy", 1, {"policy": "kv_router"}),),
    )
    with pytest.raises(ValueError, match="disagrees"):
        replay.run(_spec(bad))
    with pytest.raises(ValueError, match="execution mode 'online'"):
        replay.run(replace(_spec(), execution_mode="online"))
