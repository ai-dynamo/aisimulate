# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config adapters keep their inputs when a config uses ``ais_perf_model``.

Config adapters live outside this repository. AISim hands them plain JSON:
the engine block of a prediction, the search space and every candidate of a
recommendation, and the finished result. A router adapter, a planner adapter
and an output adapter each read a known set of keys from that JSON. The stub
adapters below read the same keys in the same order:

- router, prediction: the engine keys that are ``ForwardPassPerfModelConfig``
  fields, then the worker's ``timing`` keys on top; the worker's ``hardware``
  before the engine's.
- router, candidate: the resolved estimator under
  ``forward_pass_estimators[role].config``. A candidate without it has the
  search space loaded again and the estimator request for the role built from
  it. The stub does both for every candidate.
- planner: the deployment modes of the search space, the engine mode and each
  worker's parallelism, and the candidate's deployment mode and GPU budgets.
- output adapter: the candidate's ``hardware_sku``, ``systems_paths`` and
  ``backend``, and the search configuration saved with the result.

``ais_perf_model`` is another spelling of estimator fields those adapters
already read. Each test runs the real recommendation or prediction path twice,
once with the new key and once with the existing fields that must mean the
same, and compares everything the stubs were handed and everything they read.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import fields
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import aisim_core.systems
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.config_adapter import CONFIG_ADAPTER_API_VERSION, validate_config_adapter
from aisim.predict import _compile_prediction_adapters, _prediction_adapter_context
from aisim.recommend import _run_recommendation
from aisim.sweeper.config import SearchSpace, SmartSearchConfig
from aisim.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
from aisim.sweeper.provider import AdapterReplaySpec, AdapterSearchPlan
from aisim.sweeper.replay import ReplayReport, RunnerCapabilities
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisim_core.sdk.perf_database import load_system_spec

# Adapter owners run their suites with warnings as errors, so a warning on these paths is a failure.
pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("error")]

MODEL, SYSTEM = "Qwen/Qwen3-0.6B", "h200_sxm"
# The stubs take the names of the stack whose router and planner adapters they stand in for.
STACK = "dynamo"
NEW_KEY = "ais_perf_model"
ROLES = {"aggregated": ("aggregated",), "disaggregated": ("prefill", "decode")}
TRAFFIC = {
    "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 16},
    "load": {"type": "concurrency", "concurrency": 4},
    "stop": {"requests": 8},
}
# One trial is enough: an adapter has read all its inputs before the first replay starts.
SEARCH = {
    "optimization": {"target": "throughput", "constraints": {"max_candidate_gpus": 2}},
    "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
}
# The packaged data root written as a path: the same data as ``default``, but another authored value.
PACKAGED_ROOT = str(next(iter(aisim_core.systems.__path__)))
# One value for every estimator key a worker type may set.
SHARED = {
    "estimation_mode": "op_level",
    "fallback_policy": "deny",
    "estimator_config": {"correction": {"enabled": False}},
    "systems_paths": ["default"],
    "database_mode": "HYBRID",
    "transfer_policy": "xshape",
}


# --- Configs ---------------------------------------------------------------------------------


def _prediction(mode: str) -> dict[str, Any]:
    """A small prediction config without any estimator key."""
    engine = {
        "mode": mode,
        "model": MODEL,
        "hardware": SYSTEM,
        "backend": "vllm",
        "context_length": 256,
        "workers": {role: {} for role in ROLES[mode]},
    }
    return deepcopy({"engine": engine, "traffic": TRAFFIC})


def _recommendation(mode: str) -> dict[str, Any]:
    """The same engine and traffic as ``_prediction``, with a one-trial search."""
    return {**_prediction(mode), **deepcopy(SEARCH)}


class Case(NamedTuple):
    """One ``ais_perf_model`` value and the existing fields that must mean the same.

    A key on which every entry gives the same value belongs at engine level.
    Any other key belongs under the ``timing`` of each worker that gives it.
    """

    new: Any
    engine_keys: dict[str, Any]
    timing_keys: dict[str, dict[str, Any]]


def _with_new_key(base: dict[str, Any], value: Any) -> dict[str, Any]:
    raw = deepcopy(base)
    raw["engine"][NEW_KEY] = deepcopy(value)
    return raw


def _with_existing_fields(
    base: dict[str, Any], engine_keys: dict[str, Any], timing_keys: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """``base`` with estimator keys at engine level and under the ``timing`` of each named worker."""
    raw = deepcopy(base)
    raw["engine"].update(deepcopy(engine_keys))
    for role, keys in timing_keys.items():
        raw["engine"]["workers"][role]["timing"] = deepcopy(keys)
    return raw


def _spellings(base: dict[str, Any], case: Case) -> tuple[dict[str, Any], dict[str, Any]]:
    """``base`` written with the new key, and written with the existing fields."""
    return _with_new_key(base, case.new), _with_existing_fields(base, case.engine_keys, case.timing_keys)


def _entries(by_role: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The list form of ``ais_perf_model``: one entry for each worker type."""
    return [{"worker_type": role, **keys} for role, keys in by_role.items()]


def _cases() -> dict[str, tuple[str, Case]]:
    """Test id -> (engine mode, case)."""
    cases: dict[str, tuple[str, Case]] = {}
    for mode, roles in ROLES.items():
        cases[f"{mode}-one_config"] = (mode, Case(SHARED, SHARED, {}))
        cases[f"{mode}-agreeing_list"] = (mode, Case(_entries(dict.fromkeys(roles, SHARED)), SHARED, {}))

    every_key_differs = {
        "prefill": {"estimation_mode": "op_level", "fallback_policy": "allow", "systems_paths": ["default"]},
        "decode": {"estimation_mode": "auto", "fallback_policy": "deny", "systems_paths": [PACKAGED_ROOT]},
    }
    cases["disaggregated-every_key_differs"] = (
        "disaggregated",
        Case(_entries(every_key_differs), {}, every_key_differs),
    )
    one_key_differs = {"prefill": {"estimation_mode": "op_level"}, "decode": {"estimation_mode": "auto"}}
    cases["disaggregated-one_key_differs"] = (
        "disaggregated",
        Case(
            _entries({role: {**keys, "systems_paths": ["default"]} for role, keys in one_key_differs.items()}),
            {"systems_paths": ["default"]},
            one_key_differs,
        ),
    )
    one_role_only = {"prefill": {"estimation_mode": "op_level"}}
    cases["disaggregated-key_on_one_role_only"] = (
        "disaggregated",
        Case(_entries({**one_role_only, "decode": {}}), {}, one_role_only),
    )
    return cases


CASES = _cases()


# --- What the adapters read ------------------------------------------------------------------


def _router_prediction_read(engine: Mapping[str, Any]) -> dict[str, Any]:
    """The perf model config a router builds for the worker that takes new requests."""
    role = "prefill" if engine.get("mode") == "disaggregated" else "aggregated"
    worker = engine["workers"][role]
    known = {field.name for field in fields(ForwardPassPerfModelConfig)}
    config: dict[str, Any] = {}
    # The worker's timing keys are applied last, so they win over the engine keys.
    for source in (engine, worker.get("timing", {})):
        config.update({key: deepcopy(value) for key, value in source.items() if key in known and value is not None})
    parallelism = worker.get("parallelism", {})
    config.update(
        model=engine["model"],
        system=worker.get("hardware") or engine["hardware"],
        backend=engine["backend"],
        worker_type=role,
        tp=parallelism.get("tensor", 1),
        pp=parallelism.get("pipeline", 1),
        attention_dp=parallelism.get("attention_data", 1),
    )
    block_size = worker.get("kv_cache", {}).get("block_size")
    if block_size is not None:
        config["kv_block_size"] = block_size
    return ForwardPassPerfModelConfig(**config).to_dict()


def _router_candidate_read(sample: Mapping[str, Any], core_search_space: Mapping[str, Any]) -> dict[str, Any]:
    """The resolved estimator of a candidate, and the request a router builds when there is none."""
    role = "prefill" if sample["deployment_mode"] == "disagg" else "agg"
    resolved = sample.get("forward_pass_estimators", {}).get(role, {}).get("config")

    # A router fills in only these four search space keys and these sample keys itself. Every
    # other search space field has to load from the dump as it is.
    space = dict(core_search_space)
    space.setdefault("model_name", sample["model_name"])
    space.setdefault("hardware_sku", sample["hardware_sku"])
    space.setdefault("backend", [sample["backend"]])
    space.setdefault("backend_version", sample.get("backend_version"))
    request_sample = dict(sample)
    prefix = "prefill_" if role == "prefill" else ""
    for name in ("pp", "moe_tp", "moe_ep"):
        request_sample.setdefault(prefix + name, 1)
    request_sample.setdefault(f"{role}_block_size", None)
    request = ForwardPassEstimatorResolver(SearchSpace.model_validate(space))._request(request_sample, role)
    return {"role": role, "resolved": deepcopy(resolved), "request": request.to_dict()}


def _deployment_modes_read(core_search_space: Mapping[str, Any]) -> list[str]:
    return [str(mode) for mode in core_search_space.get("deployment_mode", ["agg", "disagg"])]


def _planner_engine_read(engine: Mapping[str, Any]) -> dict[str, Any]:
    """The deployment mode and the worker shapes a planner takes from the engine block."""
    mode = "disagg" if engine.get("mode") == "disaggregated" else "agg"
    shape: dict[str, Any] = {"deployment_mode": mode}
    for role in ("prefill", "decode") if mode == "disagg" else ("aggregated",):
        parallelism = engine["workers"][role]["parallelism"]
        prefix = "" if role == "aggregated" else f"{role}_"
        shape[f"{prefix}tp"] = int(parallelism.get("tensor", 1))
        shape[f"{prefix}attention_dp"] = int(parallelism.get("attention_data", 1))
    return shape


def _planner_candidate_read(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "deployment_mode": str(sample["deployment_mode"]),
        "gpu_budget": sample.get("gpu_budget"),
        "min_gpu_budget": sample.get("min_gpu_budget"),
    }


def _output_adapter_read(result: Any) -> dict[str, Any]:
    """What an adapter that writes deployment files takes from a finished recommendation."""
    [candidate] = result.selected_candidates
    parameters = candidate.config
    systems_paths = parameters.get("systems_paths")
    if systems_paths is not None:
        # The data roots are passed on as they are, so they must be a usable list.
        assert isinstance(systems_paths, list) and systems_paths
        assert all(isinstance(path, str) and path.strip() for path in systems_paths)
    system = load_system_spec(parameters.get("hardware_sku"), systems_paths=systems_paths)
    saved = SmartSearchConfig.model_validate(result.provenance.config)
    return {
        "hardware_sku": parameters.get("hardware_sku"),
        "systems_paths": systems_paths,
        "backend": str(parameters.get("backend", "")),
        "node": deepcopy(system.get("node")),
        "workload": saved.workload.model_dump(mode="json"),
    }


# --- Stub adapters ---------------------------------------------------------------------------


class _StubAdapter:
    """Keeps what one adapter was handed (``seen``) and what it read from that (``reads``)."""

    # The interface version adapters outside this repository declare.
    config_adapter_api_version = 3

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}
        self.reads: dict[str, Any] = {}

    def _see(self, stage: str, **mappings: Mapping[str, Any]) -> None:
        for name, mapping in mappings.items():
            self.seen[f"{stage} {name}"] = deepcopy(dict(mapping))


class StubRouterAdapter(_StubAdapter):
    name = f"{STACK}.router"
    section = "router"

    def compile_prediction(self, config, context):
        del config
        self._see("prediction", engine=context.engine, traffic=context.traffic, evaluation=context.evaluation)
        self.reads["router prediction"] = _router_prediction_read(context.engine)
        return AdapterReplaySpec(config={"perf_model_config": self.reads["router prediction"]})

    def compile_recommendation(self, config, context):
        del config
        self._see(
            "recommendation",
            engine=context.engine,
            traffic=context.traffic,
            evaluation=context.evaluation,
            optimization=context.optimization,
            core_search_space=context.sweep.core_search_space,
            workload=context.sweep.workload,
            goal=context.sweep.goal,
        )
        self.reads["router deployment modes"] = _deployment_modes_read(context.sweep.core_search_space)
        return AdapterSearchPlan(state={"core_search_space": deepcopy(dict(context.sweep.core_search_space))})

    def materialize_candidate(self, plan, selection, context):
        del selection
        self._see("candidate", sample=context.sample)
        self.reads["router candidate"] = _router_candidate_read(context.sample, plan.state["core_search_space"])
        return AdapterReplaySpec(config={"perf_model_config": self.reads["router candidate"]["resolved"]})


class StubPlannerAdapter(_StubAdapter):
    name = f"{STACK}.planner"
    section = "planner"

    def compile_prediction(self, config, context):
        del config
        self.reads["planner prediction"] = _planner_engine_read(context.engine)
        self.reads["planner prediction sla"] = deepcopy(context.evaluation.get("sla"))
        return AdapterReplaySpec(config={"sample": self.reads["planner prediction"]})

    def compile_recommendation(self, config, context):
        del config
        self.reads["planner deployment modes"] = _deployment_modes_read(context.sweep.core_search_space)
        self.reads["planner recommendation"] = _planner_engine_read(context.engine)
        return AdapterSearchPlan(state={})

    def materialize_candidate(self, plan, selection, context):
        del plan, selection
        self.reads["planner candidate"] = _planner_candidate_read(context.sample)
        return AdapterReplaySpec(config=dict(self.reads["planner candidate"]))


class _Runner:
    """Stands in for the replay. The adapters have all their inputs before it runs."""

    def run(self, spec, *, output_requirements=None):
        del spec, output_requirements
        return ReplayReport(
            metrics={
                "completed_requests": 1.0,
                "num_ttft_samples": 1.0,
                "num_tpot_samples": 1.0,
                "num_e2e_latency_samples": 1.0,
                "output_throughput_tok_s": 8.0,
                "mean_ttft_ms": 2.0,
                "mean_tpot_ms": 1.0,
                "mean_e2e_latency_ms": 4.0,
            }
        )

    def close(self) -> None:
        pass


class _RunnerFactory:
    """A module-level class, because the search sends the factory to a worker process."""

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"), ("vllm", "disagg")))

    def create(self, worker_id: int) -> _Runner:
        del worker_id
        return _Runner()


def _adapters() -> dict[str, _StubAdapter]:
    return {StubRouterAdapter.name: StubRouterAdapter(), StubPlannerAdapter.name: StubPlannerAdapter()}


# --- Running the real paths ------------------------------------------------------------------


@cache
def _recommendation_run(raw_json: str) -> dict[str, Any]:
    adapters = _adapters()
    result = _run_recommendation(
        CoreRecommendationConfig.model_validate(json.loads(raw_json)),
        adapter_configs={"router": {}, "planner": {}},
        stack=STACK,
        runner_factory=_RunnerFactory(),
        providers=adapters,
        show_progress=False,
    )
    router, planner = adapters.values()
    return {
        "seen": router.seen,
        "reads": {**router.reads, **planner.reads, "output adapter": _output_adapter_read(result)},
        "saved config": deepcopy(result.provenance.config),
    }


def _recommend(raw: dict[str, Any]) -> dict[str, Any]:
    """Run a recommendation with the stub adapters and return what they saw and read.

    One run takes a few seconds and several cases compare against the same
    existing-field config, so equal configs share one run.
    """
    return deepcopy(_recommendation_run(json.dumps(raw, sort_keys=True)))


def _predict(raw: dict[str, Any]) -> dict[str, Any]:
    """Compile a prediction's adapter sections with the stub adapters and return what they saw and read."""
    adapters = _adapters()
    replays = _compile_prediction_adapters(
        {"router": {}, "planner": {}},
        adapters,
        stack=STACK,
        context=_prediction_adapter_context(CorePredictionConfig.model_validate(raw)),
    )
    router, planner = adapters.values()
    return {
        "seen": router.seen,
        "reads": {**router.reads, **planner.reads},
        "replays": {name: replay.config for name, replay in replays.items()},
    }


def _keys(value: Any) -> Iterator[str]:
    """Every mapping key in ``value``, at any depth."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _keys(item)


# --- Existing fields only --------------------------------------------------------------------


def test_stubs_are_valid_config_adapters() -> None:
    assert CONFIG_ADAPTER_API_VERSION == 3
    for name, adapter in _adapters().items():
        assert validate_config_adapter(adapter, requested_name=name) is adapter


def test_engine_level_keys_reach_candidate_parameters_and_worker_timing_keys_do_not() -> None:
    """The two existing levels are not interchangeable, so the new key has to pick the right one."""
    base = _recommendation("aggregated")
    engine_level = _recommend(_with_existing_fields(base, SHARED, {}))
    role_level = _recommend(_with_existing_fields(base, {}, {"aggregated": SHARED}))

    assert engine_level["seen"]["candidate sample"]["systems_paths"] == ["default"]
    assert engine_level["reads"]["output adapter"]["systems_paths"] == ["default"]
    assert "systems_paths" not in role_level["seen"]["candidate sample"]
    assert role_level["reads"]["output adapter"]["systems_paths"] is None
    assert (
        engine_level["seen"]["recommendation core_search_space"]
        != role_level["seen"]["recommendation core_search_space"]
    )
    # The estimator that times the worker is the same either way. Only the level differs.
    assert engine_level["reads"]["router candidate"]["resolved"] == role_level["reads"]["router candidate"]["resolved"]


def test_worker_timing_keys_win_over_engine_keys_in_the_router_prediction_read() -> None:
    engine_keys = {"estimation_mode": "auto", "fallback_policy": "allow"}
    timing_keys = {"prefill": {"estimation_mode": "op_level"}}

    predicted = _predict(_with_existing_fields(_prediction("disaggregated"), engine_keys, timing_keys))

    engine = predicted["seen"]["prediction engine"]
    assert {key: engine[key] for key in engine_keys} == engine_keys
    assert engine["workers"]["prefill"]["timing"]["estimation_mode"] == "op_level"
    read = predicted["reads"]["router prediction"]
    assert (read["worker_type"], read["estimation_mode"], read["fallback_policy"]) == ("prefill", "op_level", "allow")


# --- The new key -----------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", CASES)
def test_recommendation_adapters_read_the_same_values(case_id: str) -> None:
    mode, case = CASES[case_id]
    new, old = (_recommend(raw) for raw in _spellings(_recommendation(mode), case))

    assert new == old
    # The new key is gone before an adapter or a saved config gets the mapping.
    assert NEW_KEY not in set(_keys(new["seen"]))
    assert NEW_KEY not in set(_keys(new["saved config"]))
    # A candidate carries data roots only when every entry agrees on them.
    roots = case.engine_keys.get("systems_paths")
    assert new["seen"]["candidate sample"].get("systems_paths") == roots
    assert new["reads"]["output adapter"]["systems_paths"] == roots
    # The saved search space loads again and gives the adapters the same input.
    reloaded = SearchSpace.model_validate(new["saved config"]["search_space"])
    assert reloaded.model_dump(mode="json") == new["seen"]["recommendation core_search_space"]


@pytest.mark.parametrize("case_id", CASES)
def test_prediction_adapters_read_the_same_values(case_id: str) -> None:
    mode, case = CASES[case_id]
    new, old = (_predict(raw) for raw in _spellings(_prediction(mode), case))

    assert new == old
    assert NEW_KEY not in set(_keys(new["seen"]))


def test_differing_entries_reach_the_router_as_worker_settings() -> None:
    mode, case = CASES["disaggregated-every_key_differs"]
    prefill = case.timing_keys["prefill"]
    # ``systems_paths`` is left out: the perf model config turns ``default`` into a directory.
    keys = ("estimation_mode", "fallback_policy")

    predicted = _predict(_with_new_key(_prediction(mode), case.new))["reads"]["router prediction"]
    assert predicted["worker_type"] == "prefill"
    assert {key: predicted[key] for key in keys} == {key: prefill[key] for key in keys}

    recommended = _recommend(_with_new_key(_recommendation(mode), case.new))
    assert recommended["seen"]["recommendation core_search_space"]["role_estimator_controls"]["prefill"] == prefill
    candidate = recommended["reads"]["router candidate"]
    assert candidate["role"] == "prefill"
    # The resolved estimator names the selected mode with ``deny``, whatever was written. The
    # written values are on the request.
    assert candidate["resolved"]["worker_type"] == "prefill"
    assert {key: candidate["request"][key] for key in keys} == {key: prefill[key] for key in keys}


# --- A state snapshot ------------------------------------------------------------------------


def _iteration(prefill_tokens: int, decode_requests: int, kv_tokens: int) -> dict[str, Any]:
    return {
        "version": 1,
        "wall_time": 0.0,
        "scheduled_requests": {
            "num_prefill_requests": int(prefill_tokens > 0),
            "sum_prefill_tokens": prefill_tokens,
            "num_decode_requests": decode_requests,
            "sum_decode_kv_tokens": kv_tokens,
        },
    }


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory) -> Path:
    """A calibrated state snapshot of an aggregated worker.

    A snapshot is usable only after calibration, so the model observes
    iterations that are 1.5 times slower than its own estimate.
    """
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=MODEL,
            system=SYSTEM,
            backend="vllm",
            worker_type="aggregated",
            estimation_mode="op_level",
            kv_block_size=16,
            estimator_config={"correction": {"mode": "fixed", "min_observations": 3}},
        )
    )
    observed = []
    for index in range(60):
        if index % 2:
            iteration = _iteration(256 * (1 + index % 8), 0, 0)
        else:
            iteration = _iteration(0, 1 + index % 32, 2048 * (1 + index % 16))
        iteration["wall_time"] = model.estimate_forward_pass_time_ms([iteration]) * 1.5 / 1000.0
        observed.append([iteration])
    model.tune_with_fpms(observed)
    path = tmp_path_factory.mktemp("snapshot") / "aggregated"
    model.save_state_snapshot(path, include_perf_db=False)
    return path


def test_a_snapshot_entry_reads_like_a_snapshot_in_the_worker_timing(snapshot: Path) -> None:
    reference = {"state_snapshot": {"path": str(snapshot)}}
    case = Case({"from": reference}, {}, {"aggregated": {"estimator_config": reference}})

    new, old = (_recommend(raw) for raw in _spellings(_recommendation("aggregated"), case))
    assert new == old
    assert NEW_KEY not in set(_keys(new["seen"]))
    assert NEW_KEY not in set(_keys(new["saved config"]))
    # A snapshot belongs to one worker, so it is a worker setting and no candidate parameter.
    controls = new["seen"]["recommendation core_search_space"]["role_estimator_controls"]["agg"]
    assert controls["estimator_config"] == reference
    assert "systems_paths" not in new["seen"]["candidate sample"]
    request = new["reads"]["router candidate"]["request"]
    assert request["estimator_config"]["state_snapshot"]["path"] == str(snapshot)

    new, old = (_predict(raw) for raw in _spellings(_prediction("aggregated"), case))
    assert new == old
    assert NEW_KEY not in set(_keys(new["seen"]))
    read = new["reads"]["router prediction"]
    assert read["estimator_config"]["state_snapshot"]["path"] == str(snapshot)
