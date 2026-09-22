# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aisimulate.config import CoreRecommendationConfig
from aisimulate.config.common import ResourceConfig, split_config_sections
from aisimulate.resource_scheduler import InterruptedEvaluation, evaluate_waves
from aisimulate.resources import (
    GB,
    GuardedRunnerFactory,
    HostResources,
    ResourceEstimate,
    ResourceLimitError,
    workload_bounds,
)


@dataclass
class _BudgetFactory:
    capacity: int = 2
    admitted: list[list[int]] = field(default_factory=list)

    def admit_wave(self, specs):
        if sum(item["cost"] for item in specs) > self.capacity:
            raise ResourceLimitError("combined wave does not fit")
        self.admitted.append([item["id"] for item in specs])
        return {"status": "admitted"}

    def live_pressure(self):
        return None


def _init(factory):
    pass


def _evaluate(spec):
    if spec.get("refuse"):
        raise ResourceLimitError("worker guard refused allocation")
    return {"id": spec["id"]}


def test_admission_splits_waves_and_skips_only_oversized_candidates():
    factory = _BudgetFactory()
    specs = [{"id": i, "cost": cost} for i, cost in enumerate([1, 1, 3, 1])]
    results = dict(evaluate_waves(specs, factory=factory, initializer=_init, evaluate=_evaluate, workers=4, timeout=10))
    assert results[0] == {"id": 0}
    assert results[1] == {"id": 1}
    assert isinstance(results[2], InterruptedEvaluation)
    assert results[2].resource_limited
    assert results[3] == {"id": 3}
    assert factory.admitted == [[0, 1], [3]]


def _reported_replay_sentinel(spec):
    assert spec.concurrency == 8, "oversized reported candidate reached replay"
    return spec.concurrency


@pytest.mark.parametrize("concurrencies", [(64_512,), (8, 64_512), (64_512, 8)])
def test_reported_mac_sweep_refuses_largest_and_keeps_fitting_candidates(monkeypatch, concurrencies):
    from aisimulate import resources

    path = Path(__file__).parent / "e2e/configs/resource_safety/reported-mac-sweep.yaml"
    raw = yaml.safe_load(path.read_text())
    assert hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == (
        "af1a9d53df52c5eab2c07f9be20b69457e51ddd4c43933c42bcca7e489698577"
    )
    core, _ = split_config_sections(raw, command="recommend")
    specs = []
    for concurrency in concurrencies:
        core["traffic"]["load"]["concurrency"] = concurrency
        bounds = workload_bounds(CoreRecommendationConfig.model_validate(core))
        specs.append(SimpleNamespace(concurrency=concurrency, workload=bounds))

    monkeypatch.setattr(resources, "discover_host", lambda: HostResources(32 * GB, 16 * GB, 8))
    factory = GuardedRunnerFactory(object(), "dynamo", ResourceConfig(memory_limit_gb=2.0, cpu_limit=1))
    results = dict(
        evaluate_waves(
            specs, factory=factory, initializer=_init, evaluate=_reported_replay_sentinel, workers=8, timeout=10
        )
    )
    assert len(results) == len(specs)
    for index, concurrency in enumerate(concurrencies):
        result = results[index]
        if concurrency == 64_512:
            assert isinstance(result, InterruptedEvaluation)
            assert result.resource_limited
            assert result.metadata["estimate"]["request_count"] == 6_451_200
            assert result.metadata["estimate"]["allocation_model"] == "dynamo-eager-u32-v1"
        else:
            assert result == concurrency


def test_checkpoint_failure_stops_before_workers_without_reclassifying_candidates(monkeypatch):
    from aisimulate import resource_scheduler as scheduler

    factory = _BudgetFactory()
    failure = ResourceLimitError("execution evidence exceeds the bounded checkpoint budget")

    def fail_checkpoint(event, plan):
        raise failure

    def unexpected_pool(**kwargs):
        pytest.fail("evidence refusal must precede worker creation")

    monkeypatch.setattr(scheduler, "checkpoint", fail_checkpoint)
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", unexpected_pool)
    specs = [{"id": i, "cost": 1} for i in range(2)]
    with pytest.raises(ResourceLimitError) as caught:
        next(evaluate_waves(specs, factory=factory, initializer=_init, evaluate=_evaluate, workers=2, timeout=10))
    assert caught.value is failure
    assert factory.admitted == [[0, 1]]


def test_resource_retries_are_bounded_and_preserve_completed_work():
    factory = _BudgetFactory()
    specs = [{"id": 0, "cost": 1}, {"id": 1, "cost": 1, "refuse": True}]
    results = list(evaluate_waves(specs, factory=factory, initializer=_init, evaluate=_evaluate, workers=2, timeout=10))
    assert dict(results)[0] == {"id": 0}
    assert dict(results)[1].resource_limited
    assert len(results) == 2
    assert sum(1 in batch for batch in factory.admitted) == 3
    assert all(len(batch) == 1 for batch in factory.admitted[1:])


class _EstimateFactory:
    def estimate_host_resources(self, workload, *, concurrency=None):
        return ResourceEstimate("test-v1", 1, 0, 0, 6 * GB)


def test_whole_wave_reserves_sum_against_one_live_snapshot(monkeypatch):
    from aisimulate import resources

    monkeypatch.setattr(resources, "discover_host", lambda: HostResources(16 * GB, 9 * GB, 8))
    factory = GuardedRunnerFactory(_EstimateFactory(), "custom", ResourceConfig())
    spec = SimpleNamespace(workload={}, concurrency=1)
    assert factory.admit_wave([spec])["status"] == "admitted"
    with pytest.raises(ResourceLimitError) as caught:
        factory.admit_wave([spec, spec])
    assert caught.value.plan["required_bytes"] == 12 * GB
    assert caught.value.plan["available_bytes"] < 8 * GB
    monkeypatch.setattr(resources, "discover_host", lambda: HostResources(16 * GB, 6 * GB, 8))
    with pytest.raises(ResourceLimitError):
        factory.admit_wave([spec])


def test_live_pressure_retries_only_unfinished_work_after_cleanup(monkeypatch):
    from concurrent.futures import Future

    from aisimulate import resource_scheduler as scheduler

    alive = []
    waves = []

    class Factory(_BudgetFactory):
        def admit_wave(self, specs):
            assert not alive, "old workers must be reaped before admission"
            return super().admit_wave(specs)

        def live_pressure(self):
            return {"reason": "pressure"} if len(waves) == 1 else None

    class Pool:
        def __init__(self, **kwargs):
            alive.append(self)
            waves.append([])

        def submit(self, evaluate, spec):
            waves[-1].append(spec["id"])
            future = Future()
            if len(waves) > 1 or spec["id"] == 0:
                future.set_result({"id": spec["id"]})
            return future

    def stop(pool):
        alive.remove(pool)

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(scheduler, "terminate_pool", stop)
    monkeypatch.setattr(scheduler, "close_pool", stop)
    specs = [{"id": i, "cost": 1} for i in range(2)]
    results = list(
        evaluate_waves(specs, factory=Factory(), initializer=_init, evaluate=_evaluate, workers=2, timeout=2)
    )
    assert results == [(0, {"id": 0}), (1, {"id": 1})]
    assert waves == [[0, 1], [1]]
    assert not alive


def _hung_initializer(factory):
    import time

    time.sleep(30)


def test_worker_initialization_is_bounded_without_a_replay_timeout():
    factory = _BudgetFactory()
    factory.policy = ResourceConfig(initialization_timeout_seconds=0.2)
    results = dict(
        evaluate_waves(
            [{"id": 0, "cost": 1}],
            factory=factory,
            initializer=_hung_initializer,
            evaluate=_evaluate,
            workers=1,
            timeout=None,
        )
    )
    assert isinstance(results[0], InterruptedEvaluation)
    assert results[0].reason == "worker initialization timed out"


def test_readiness_counts_actual_workers_when_the_pool_reuses_one(monkeypatch):
    from concurrent.futures import Future
    from pathlib import Path

    from aisimulate import resource_scheduler as scheduler

    elapsed = [0.0]
    finished = Future()
    finished.set_result({"id": 0})
    remaining = Future()

    class Pool:
        def __init__(self, **kwargs):
            self._processes = {123: object()}
            Path(kwargs["initargs"][2], "123").touch()

        def submit(self, evaluate, spec):
            return finished if spec["id"] == 0 else remaining

    def advance(futures, **kwargs):
        # Both jobs use one ready worker. The second runs longer than the
        # initialization deadline but remains inside its replay timeout.
        elapsed[0] += 0.2
        if elapsed[0] > 0.6 and not remaining.done():
            remaining.set_result({"id": 1})
        return {future for future in futures if future.done()}, set()

    factory = _BudgetFactory()
    factory.policy = ResourceConfig(initialization_timeout_seconds=0.1)
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(scheduler, "close_pool", lambda pool: None)
    monkeypatch.setattr(scheduler, "terminate_pool", lambda pool: None)
    monkeypatch.setattr(scheduler, "wait", advance)
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: elapsed[0])
    specs = [{"id": i, "cost": 1} for i in range(2)]
    results = dict(evaluate_waves(specs, factory=factory, initializer=_init, evaluate=_evaluate, workers=2, timeout=2))
    assert results == {0: {"id": 0}, 1: {"id": 1}}
