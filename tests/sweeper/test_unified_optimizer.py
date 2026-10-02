# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import ClassVar

import pytest

import aisimulate.sweeper.search as search_module
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities
from aisimulate.sweeper.sampler import (
    InvalidSuggestionError,
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
    replica = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
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


def test_global_trial_budget_is_split_across_branches(monkeypatch) -> None:
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(search_module, "enumerate_branches", lambda *args, **kwargs: _branches())
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, systems_paths=None: "test")
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
    monkeypatch.setattr(search_module, "enumerate_branches", lambda *args, **kwargs: _branches())
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, systems_paths=None: "test")
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
    monkeypatch.setattr(search_module, "enumerate_branches", lambda *args, **kwargs: _branches())
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, systems_paths=None: "test")
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
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, systems_paths=None: "test")
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


@pytest.mark.parametrize(
    ("max_trials", "error_type"),
    [(None, InvalidSuggestionError), (8, InvalidSuggestionError), (8, KeyError)],
)
def test_sampler_failure_recovery_is_narrow_and_preserves_best_results(
    monkeypatch, caplog, max_trials, error_type
) -> None:
    calls = []
    observed = []
    events = []

    class FailingSampler(_CountingSampler):
        def suggest(self, count):
            calls.append(self.branch.deployment_mode)
            if len(calls) > 2:
                raise error_type("missing parameters: agg_max_num_batched_tokens")
            suggestions = super().suggest(count)
            # Distinct candidates even in the legacy branch-major order.
            for suggestion in suggestions:
                if self.branch.deployment_mode == "agg":
                    suggestion.selection["agg_max_num_seqs"] += len(calls)
            return suggestions

        def observe(self, suggestion, metrics):
            observed.append(metrics)

        def observe_infeasible(self, suggestion, reason):
            pytest.fail(f"unexpected infeasible observation: {reason}")

    class ScoredRunner(_Runner):
        def run(self, spec):
            report = super().run(spec)
            report.metrics["output_throughput_tok_s"] = float(_Runner.runs * 10)
            return report

    runner = ScoredRunner()
    monkeypatch.setattr(_Factory, "create", lambda self, worker_id: runner)
    monkeypatch.setattr(search_module, "enumerate_branches", lambda *args, **kwargs: _branches())
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, **kwargs: "test")
    monkeypatch.setattr(
        "aisimulate.supervision.checkpoint",
        lambda event, value: events.append((event, value)),
    )
    _Runner.runs = 0
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "model",
                "hardware_sku": "hardware",
                "deployment_mode": ["agg", "disagg"],
            },
            "workload": {"isl": 8, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
            "sweep": {
                "max_rounds": 4,
                "parallel_evals": 1,
                "candidates_per_round": 1,
                "max_eval_seconds": None,
                "max_trials": max_trials,
            },
        }
    )
    sweeper = search_module.Sweeper(runner_factory=_Factory(), sampler_factory=FailingSampler, show_progress=False)
    if error_type is KeyError:
        # A programming error must propagate even when partial results exist.
        with pytest.raises(KeyError, match="agg_max_num_batched_tokens"):
            sweeper.run(config, top_n=1)
        return
    result = sweeper.run(config, top_n=1)

    assert len(calls) == 3  # No retry and no later branch is searched.
    assert _Runner.runs == len(observed) == result.counts.feasible == result.counts.evaluated == 2
    assert result.counts.failed == result.counts.infeasible == 0
    assert result.selected_candidate_ids == ["candidate-000002"]
    assert result.selected_candidates[0].score == 20.0
    assert "agg_max_num_batched_tokens" in caplog.text
    assert "search is incomplete" in caplog.text
    assert events[-1][0] == "optimizer_stopped"


def test_seeded_random_sampler_is_deterministic() -> None:
    branch = _branches()[0]
    first = RandomBranchSampler(branch, seed=11).suggest(4)
    second = RandomBranchSampler(branch, seed=11).suggest(4)

    assert [item.selection for item in first] == [item.selection for item in second]
    assert [item.parallel_config for item in first] == [item.parallel_config for item in second]


def test_seeded_bayesian_sampler_is_deterministic() -> None:
    branch = _branches()[0]
    branch.knob_choices["agg_max_num_seqs"] = [256, 512, 1024]
    first = SeededBayesianBranchSampler(branch, objectives=None, seed=13).suggest(2)
    second = SeededBayesianBranchSampler(branch, objectives=None, seed=13).suggest(2)

    assert [item.selection for item in first] == [item.selection for item in second]


def test_seeded_bayesian_sampler_clears_compilation_cache_between_batches() -> None:
    import jax

    branch = _branches()[0]
    branch.knob_choices["agg_max_num_seqs"] = [256, 512, 1024]
    sampler = SeededBayesianBranchSampler(branch, objectives=None, seed=13)
    seed = sampler.suggest(1)[0]
    sampler.observe(seed, {"objective": 1.0})

    traces = []

    @jax.jit
    def compiled(value):
        traces.append(None)
        return value + 1

    assert int(compiled(1)) == 2
    assert len(traces) == 1
    for batch in range(2):
        # Repeating the same signature reuses the executable until the next ask.
        assert int(compiled(1)) == 2
        assert len(traces) == batch + 1
        suggestion = sampler.suggest(1)[0]
        assert suggestion.selection["agg_max_num_seqs"] in [256, 512, 1024]
        assert int(compiled(1)) == 2
        assert len(traces) == batch + 2
        sampler.observe(suggestion, {"objective": float(batch + 2)})


def test_candidate_timeout_applies_with_parallelism_one(monkeypatch) -> None:
    _CountingSampler.created = []
    _CountingSampler.suggestion_batches = []
    _CountingSampler.suggested = 0
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda *args, **kwargs: [_branches()[0]],
    )
    monkeypatch.setattr(search_module, "resolve_backend_version", lambda *args, systems_paths=None: "test")
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


@pytest.fixture(autouse=True)
def _isolate_estimator_data_for_orchestration(monkeypatch):
    # These tests use synthetic models/runners. Native construction is exercised
    # by the estimator contract tests and CLI round trips.
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    monkeypatch.setattr(ForwardPassEstimatorResolver, "resolve_candidate", lambda self, sample: {})
