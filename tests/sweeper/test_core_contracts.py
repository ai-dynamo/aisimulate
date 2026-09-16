# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the Dynamo-neutral adapter and replay contracts."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from enum import Enum

import pytest

from aisimulate import sweeper
from aisimulate.sweeper.provider import (
    API_VERSION,
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    RuntimeHookSpec,
    SearchSpaceFragment,
    SweepConfigProvider,
    SweepContext,
)
from aisimulate.sweeper.replay import (
    REPLAY_SPEC_API_VERSION,
    BackendDeploymentSpec,
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
    canonical_json,
    validate_json_value,
)


def _deployment(*, mode: str = "agg", backend: str = "vllm"):
    return BackendDeploymentSpec(
        deployment_mode=mode,
        backend=backend,
        backend_version="0.1",
        parallel_config={"tp": 2, "replicas": 1},
        agg_engine_args={"max_num_seqs": 256},
        num_workers=1,
    )


def _planner_hook(*, version: int = 1):
    return RuntimeHookSpec(
        provider="dynamo.planner",
        kind="scaling",
        api_version=version,
        config={"interval_seconds": 180},
    )


def _replay_spec(*, hook: RuntimeHookSpec | None = None):
    adapter = (
        {} if hook is None else {"dynamo.planner": AdapterReplaySpec(config={"enabled": True}, runtime_hooks=(hook,))}
    )
    return ReplaySpec(
        backend_deployment=_deployment(),
        workload={"isl": 128, "osl": 32, "request_rate": 2.0},
        goal={"target": "throughput"},
        concurrency=4,
        adapters=adapter,
    )


def test_contracts_pickle_and_canonical_json_round_trip():
    hook = _planner_hook()
    fragment = SearchSpaceFragment(
        choices_by_branch={"agg": {"scaling_policy": ["disabled", "load"]}},
        float_ranges_by_branch={"agg": {"sensitivity": (0.0, 1.0)}},
    )
    search_plan = AdapterSearchPlan(
        fragment=fragment,
        state={"winner_by_interval": {"180": "constant"}},
        diagnostics={"loss": 0.25},
        potential_runtime_hooks=(hook,),
    )
    deployment = _deployment()
    values = [
        SweepContext(
            core_search_space={"backend": ["vllm"]},
            workload={"request_rate": 2.0},
            goal={"target": "throughput"},
            show_progress=False,
        ),
        CandidateContext(
            sample={"backend": "vllm"},
            backend_deployment=deployment,
            concurrency=4,
        ),
        fragment,
        search_plan,
        AdapterReplaySpec(config={"enabled": True}, runtime_hooks=(hook,)),
        deployment,
        _replay_spec(hook=hook),
        ReplayReport(metrics={"output_throughput_tok_s": 10.0}),
        ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
        RunnerCapabilities(
            supported_backend_topologies=(("vllm", "agg"),),
            supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
        ),
    ]

    for value in values:
        assert pickle.loads(pickle.dumps(value)) == value
        assert json.loads(canonical_json(value)) is not None


def test_canonical_json_is_stable_and_strict():
    left = {"z": [2, 1], "a": {"second": 2, "first": 1}}
    right = {"a": {"first": 1, "second": 2}, "z": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":{"first":1,"second":2},"z":[2,1]}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"metric": math.nan})
    with pytest.raises(TypeError, match="string mapping keys"):
        canonical_json({1: "not allowed"})
    with pytest.raises(TypeError, match="not supported"):
        canonical_json(object())


def test_replay_output_requirements_validate_enabled_telemetry_interval():
    assert ReplayOutputRequirements().capture_telemetry is False
    assert (
        ReplayOutputRequirements(
            capture_telemetry=True,
            telemetry_sample_interval_ms=250.0,
        ).telemetry_sample_interval_ms
        == 250.0
    )

    for invalid in (0.0, -1.0, math.inf, math.nan, True, "one second"):
        with pytest.raises(
            ValueError,
            match="telemetry_sample_interval_ms must be finite and positive",
        ):
            ReplayOutputRequirements(
                capture_telemetry=True,
                telemetry_sample_interval_ms=invalid,  # type: ignore[arg-type]
            )


def test_adapter_payload_json_validation_does_not_normalize_python_objects():
    @dataclass
    class PythonObject:
        value: int

    class StringEnum(str, Enum):
        VALUE = "value"

    validate_json_value({"nested": [None, True, 1, 1.0, "value"]})
    for value in (PythonObject(1), StringEnum.VALUE, (1, 2)):
        with pytest.raises(TypeError, match="non-JSON value"):
            validate_json_value({"nested": value})
    with pytest.raises(ValueError, match="finite JSON numbers"):
        validate_json_value({"nested": math.inf})


def test_replay_spec_collects_runtime_hooks_in_adapter_order():
    planner = _planner_hook()
    router = RuntimeHookSpec("dynamo.router", "placement", 2, {})
    spec = ReplaySpec(
        backend_deployment=_deployment(),
        workload={},
        goal={},
        adapters={
            "dynamo.router": AdapterReplaySpec(runtime_hooks=(router,)),
            "dynamo.planner": AdapterReplaySpec(runtime_hooks=(planner,)),
        },
    )

    assert spec.runtime_hooks == (router, planner)


def test_runner_capabilities_accept_supported_spec_and_wildcards():
    hook = _planner_hook()
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "*"),),
        supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
    )

    assert capabilities.supports_backend_topology("vllm", "agg")
    assert not capabilities.supports_backend_topology("sglang", "agg")
    assert capabilities.supports_hook(hook)
    capabilities.require_compatible(_replay_spec(hook=hook))


def test_runner_capabilities_require_explicit_online_support():
    spec = _replay_spec()
    spec = ReplaySpec(
        backend_deployment=spec.backend_deployment,
        workload=spec.workload,
        goal=spec.goal,
        execution_mode="online",
        concurrency=spec.concurrency,
        adapters=spec.adapters,
    )
    offline = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "agg"),),
    )
    with pytest.raises(ValueError, match="execution mode 'online'"):
        offline.require_compatible(spec)

    online = RunnerCapabilities(
        supported_execution_modes=("offline", "online"),
        supported_backend_topologies=(("vllm", "agg"),),
    )
    online.require_compatible(spec)


def test_runner_capabilities_reject_spec_version_backend_and_hook():
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "agg"),),
        supported_hooks=(HookCapability("dynamo.planner", "scaling", 1),),
    )
    wrong_version = ReplaySpec(
        backend_deployment=_deployment(),
        workload={},
        goal={},
        api_version=REPLAY_SPEC_API_VERSION + 1,
    )
    with pytest.raises(ValueError, match="ReplaySpec API version"):
        capabilities.require_compatible(wrong_version)
    with pytest.raises(ValueError, match="runner version 1"):
        capabilities.require_replay_spec_version(REPLAY_SPEC_API_VERSION + 1)

    wrong_backend = ReplaySpec(backend_deployment=_deployment(backend="sglang"), workload={}, goal={})
    with pytest.raises(ValueError, match="sglang.*agg"):
        capabilities.require_compatible(wrong_backend)

    with pytest.raises(ValueError, match=r"dynamo\.planner:scaling@2"):
        capabilities.require_compatible(_replay_spec(hook=_planner_hook(version=2)))


def test_sweep_config_provider_protocol_is_structural():
    class Adapter:
        name = "example"
        api_version = API_VERSION

        def generate_search_space(self, search_spec, context):
            return AdapterSearchPlan()

        def materialize_replay(self, plan, selection, context):
            return AdapterReplaySpec()

    assert isinstance(Adapter(), SweepConfigProvider)


def test_public_contract_versions_start_at_one():
    assert API_VERSION == 1
    assert REPLAY_SPEC_API_VERSION == 1


def test_lazy_exports_are_listed_in_public_api():
    assert set(sweeper._LAZY_EXPORTS).issubset(sweeper.__all__)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_replay_report_rejects_nonfinite_ordinary_metrics(value):
    with pytest.raises(ValueError, match="must be finite"):
        ReplayReport(metrics={"output_throughput_tok_s": value})


@pytest.mark.parametrize(
    "metrics",
    [
        {"power_w": 500.0, "power_coverage": 0.42},
        {"power_w": 500.0, "power_coverage": None},
        {"power_w": None, "power_coverage": 1.01},
    ],
)
def test_replay_report_rejects_invalid_power_pairs(metrics):
    with pytest.raises(ValueError, match="power_"):
        ReplayReport(metrics=metrics)


@pytest.mark.parametrize(
    "metrics",
    [
        {"power_w": 500.0, "power_coverage": 0.9},
        {"power_w": None, "power_coverage": 0.42},
        {"power_w": None, "power_coverage": None},
    ],
)
def test_replay_report_preserves_valid_power_availability(metrics):
    assert ReplayReport(metrics=metrics).metrics == metrics


@pytest.mark.parametrize("power_fields", [{}, {"power_coverage": 0.42}, {"power_w": None}])
def test_replay_report_materializes_power_fields_without_mutating_input(power_fields):
    metrics = {"output_throughput_tok_s": 10.0, **power_fields}
    original = metrics.copy()
    report = ReplayReport(metrics=metrics)
    assert report.metrics == {
        "output_throughput_tok_s": 10.0,
        "power_w": None,
        "power_coverage": power_fields.get("power_coverage"),
    }
    assert metrics == original
    assert json.loads(canonical_json(report))["metrics"] == report.metrics
