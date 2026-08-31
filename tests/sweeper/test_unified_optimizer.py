# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import ClassVar

import aisimulate.sweeper.search as search_module
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.replay import (
    ForwardPassEstimatorSpec,
    ReplayReport,
    RunnerCapabilities,
)
from aisimulate.sweeper.sampler import (
    RandomBranchSampler,
    SeededBayesianBranchSampler,
    Suggestion,
)
from aisimulate.sweeper.search_space import BranchSpace


class _Runner:
    runs: ClassVar[int] = 0

    def run(self, spec):
        del spec
        _Runner.runs += 1
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": 10.0,
                "gpu_hours": 1.0,
                "duration_ms": 3_600_000.0,
            }
        )

    def close(self):
        pass


class _Factory:
    def capabilities(self):
        return RunnerCapabilities(
            supported_backend_topologies=(
                ("vllm", "agg"),
                ("vllm", "disagg"),
            )
        )

    def create(self, worker_id):
        del worker_id
        return _Runner()


class _SlowRunner(_Runner):
    def run(self, spec):
        del spec
        time.sleep(1)
        return ReplayReport(metrics={"output_throughput_tok_s": 1.0})


class _SlowFactory(_Factory):
    def create(self, worker_id):
        del worker_id
        return _SlowRunner()


class _CountingSampler:
    created: ClassVar[list[tuple[str, str | None, int | None]]] = []
    suggestion_batches: ClassVar[list[tuple[str, int]]] = []
    suggested: ClassVar[int] = 0

    def __init__(self, branch, study_id, objectives=None, algorithm=None, seed=None):
        del study_id, objectives
        self.branch = branch
        self.created.append((branch.deployment_mode, algorithm, seed))

    def suggest(self, count):
        self.__class__.suggested += count
        self.__class__.suggestion_batches.append((self.branch.deployment_mode, count))
        if self.branch.deployment_mode == "agg":
            selection = {
                "deployment_mode": "agg",
                "backend": "vllm",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": 256,
            }
        else:
            selection = {
                "deployment_mode": "disagg",
                "backend": "vllm",
                "prefill_max_num_batched_tokens": 8192,
                "prefill_max_num_seqs": 1,
                "decode_max_num_batched_tokens": 8192,
                "decode_max_num_seqs": 256,
            }
        return [
            Suggestion(
                selection=dict(selection),
                parallel_config=self.branch.parallel_configs[0],
                handle=None,
            )
            for _ in range(count)
        ]

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        del suggestion, reason


def _branches():
    replica = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )
    disagg = DisaggParallelConfig(prefill=replica, decode=replica)
    return [
        BranchSpace(
            deployment_mode="agg",
            parallel_configs=(replica,),
            supported_backends={replica: frozenset({"vllm"})},
            knob_choices={
                "backend": ["vllm"],
                "agg_max_num_batched_tokens": [8192],
                "agg_max_num_seqs": [256],
            },
        ),
        BranchSpace(
            deployment_mode="disagg",
            parallel_configs=(disagg,),
            supported_backends={disagg: frozenset({"vllm"})},
            knob_choices={
                "backend": ["vllm"],
                "prefill_max_num_batched_tokens": [8192],
                "prefill_max_num_seqs": [1],
                "decode_max_num_batched_tokens": [8192],
                "decode_max_num_seqs": [256],
            },
        ),
    ]


class _StaticResolver:
    def __init__(self, specs):
        self.specs = specs

    def resolve_candidate(self, sample):
        roles = (
            ("agg",) if sample["deployment_mode"] == "agg" else ("prefill", "decode")
        )
        return {role: self.specs[sample["backend"]] for role in roles}


def _forward_pass_estimator_resolver(search_space):
    specs = {
        backend: ForwardPassEstimatorSpec(
            config={
                "model": search_space.model_name,
                "system": search_space.hardware_sku,
                "backend": backend,
                "backend_version": "test",
                "database_mode": "SILICON",
                "transfer_policy": ["xshape", "xquant", "xprofile", "xop"],
                "forward_model": "op_level",
                "systems_paths": ["/systems"],
            },
            diagnostics={"provenance": {"selected_systems_root": "/systems"}},
        )
        for backend in search_space.backend
    }
    return _StaticResolver(specs)


def test_global_trial_budget_is_split_across_branches(monkeypatch) -> None:
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module, "enumerate_branches", lambda *args, **kwargs: _branches()
    )
    monkeypatch.setattr(
        search_module,
        "ForwardPassEstimatorResolver",
        _forward_pass_estimator_resolver,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["agg", "disagg"],
            },
            "workload": {
                "isl": 8,
                "osl": 2,
                "concurrency": 1,
                "num_request_ratio": 1,
            },
            "sweep": {
                "max_rounds": 3,
                "parallel_evals": 1,
                "candidates_per_round": 2,
                "max_eval_seconds": None,
                "max_trials": 3,
                "algorithm": "random",
                "seed": 7,
            },
        }
    )

    search_module.Sweeper(
        runner_factory=_Factory(),
        sampler_factory=_CountingSampler,
        show_progress=False,
    ).run(config)

    assert _CountingSampler.suggested == 3
    assert _CountingSampler.created == [
        ("agg", "random", 7),
        ("disagg", "random", 8),
    ]


def test_global_trial_budget_runs_branch_batches_round_robin(monkeypatch) -> None:
    _Runner.runs = 0
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module, "enumerate_branches", lambda *args, **kwargs: _branches()
    )
    monkeypatch.setattr(
        search_module,
        "ForwardPassEstimatorResolver",
        _forward_pass_estimator_resolver,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["agg", "disagg"],
            },
            "workload": {
                "isl": 8,
                "osl": 2,
                "concurrency": 1,
                "num_request_ratio": 1,
            },
            "sweep": {
                "max_rounds": 4,
                "parallel_evals": 1,
                "candidates_per_round": 1,
                "max_eval_seconds": None,
                "max_trials": 4,
                "algorithm": "random",
                "seed": 7,
            },
        }
    )

    search_module.Sweeper(
        runner_factory=_Factory(),
        sampler_factory=_CountingSampler,
        show_progress=False,
    ).run(config)

    assert _CountingSampler.suggested == 4
    assert _CountingSampler.suggestion_batches == [
        ("agg", 1),
        ("disagg", 1),
        ("agg", 1),
        ("disagg", 1),
    ]
    # The second suggestion for each branch is a cache hit. It consumes the
    # public trial budget but does not launch another replay.
    assert _Runner.runs == 2


def test_legacy_rounds_remain_branch_major_without_max_trials(monkeypatch) -> None:
    _Runner.runs = 0
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module, "enumerate_branches", lambda *args, **kwargs: _branches()
    )
    monkeypatch.setattr(
        search_module,
        "ForwardPassEstimatorResolver",
        _forward_pass_estimator_resolver,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["agg", "disagg"],
            },
            "workload": {
                "isl": 8,
                "osl": 2,
                "concurrency": 1,
                "num_request_ratio": 1,
            },
            "sweep": {
                "max_rounds": 2,
                "parallel_evals": 1,
                "candidates_per_round": 1,
                "max_eval_seconds": None,
            },
        }
    )

    search_module.Sweeper(
        runner_factory=_Factory(),
        sampler_factory=_CountingSampler,
        show_progress=False,
    ).run(config)

    modes = [mode for mode, _count in _CountingSampler.suggestion_batches]
    first_disagg = modes.index("disagg")
    assert all(mode == "agg" for mode in modes[:first_disagg])
    assert all(mode == "disagg" for mode in modes[first_disagg:])
    assert _Runner.runs == 2


def test_branch_seed_is_stable_when_branch_order_changes(monkeypatch) -> None:
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda *args, **kwargs: list(reversed(_branches())),
    )
    monkeypatch.setattr(
        search_module,
        "ForwardPassEstimatorResolver",
        _forward_pass_estimator_resolver,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["disagg", "agg"],
            },
            "workload": {
                "isl": 8,
                "osl": 2,
                "concurrency": 1,
                "num_request_ratio": 1,
            },
            "sweep": {
                "max_rounds": 1,
                "parallel_evals": 1,
                "candidates_per_round": 1,
                "max_eval_seconds": None,
                "max_trials": 2,
                "algorithm": "random",
                "seed": 7,
            },
        }
    )

    search_module.Sweeper(
        runner_factory=_Factory(),
        sampler_factory=_CountingSampler,
        show_progress=False,
    ).run(config)

    assert _CountingSampler.created == [
        ("disagg", "random", 8),
        ("agg", "random", 7),
    ]


def test_seeded_random_sampler_is_deterministic() -> None:
    branch = _branches()[0]
    first = RandomBranchSampler(branch, seed=11).suggest(4)
    second = RandomBranchSampler(branch, seed=11).suggest(4)

    assert [item.selection for item in first] == [item.selection for item in second]
    assert [item.parallel_config for item in first] == [
        item.parallel_config for item in second
    ]


def test_seeded_bayesian_sampler_is_deterministic() -> None:
    branch = _branches()[0]
    branch.knob_choices["agg_max_num_seqs"] = [256, 512, 1024]
    first = SeededBayesianBranchSampler(branch, objectives=None, seed=13).suggest(2)
    second = SeededBayesianBranchSampler(branch, objectives=None, seed=13).suggest(2)

    assert [item.selection for item in first] == [item.selection for item in second]


def test_candidate_timeout_applies_with_parallelism_one(monkeypatch) -> None:
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda *args, **kwargs: [_branches()[0]],
    )
    monkeypatch.setattr(
        search_module,
        "ForwardPassEstimatorResolver",
        _forward_pass_estimator_resolver,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["agg"],
            },
            "workload": {
                "isl": 8,
                "osl": 2,
                "concurrency": 1,
                "num_request_ratio": 1,
            },
            "sweep": {
                "max_rounds": 1,
                "parallel_evals": 1,
                "candidates_per_round": 1,
                "max_eval_seconds": 0.01,
                "max_trials": 1,
                "algorithm": "random",
            },
        }
    )

    result = search_module.Sweeper(
        runner_factory=_SlowFactory(),
        sampler_factory=_CountingSampler,
        show_progress=False,
    ).run(config)

    assert result.selected_candidates == []
    assert _CountingSampler.suggested == 1
