# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD integration with the generic Sweeper branch, sampler, and replay contract."""

import pytest
from pydantic import ValidationError

from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.afd_parallel import (
    AFDInfeasible,
    AFDParallelConfig,
    AFDReasonCategory,
)
from aisimulate.sweeper.afd_perfmodel import AFDLayerTimes
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.model_hw import ModelHardware, NoViableParallelConfig
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplayReport, ReplaySpec, RunnerCapabilities
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import RandomBranchSampler
from aisimulate.sweeper.search import Sweeper
from aisimulate.sweeper.search_space import RunnerIncompatibleError, enumerate_branches


def _model_hardware() -> ModelHardware:
    return ModelHardware(
        model_name="example/model",
        hardware_sku="example_sku",
        backend="vllm",
        is_moe=False,
        mla=False,
        enable_wideep=False,
        weight_bytes=1,
        vram_per_gpu=80,
        gpus_per_node=4,
        max_context=4096,
        num_experts=0,
    )


def _topology(*, n_a_nodes: int = 1, n_f_nodes: int = 1) -> dict:
    return {
        "n_a_nodes": n_a_nodes,
        "n_f_nodes": n_f_nodes,
        "tp_a": 2,
        "a_batch_size": 16,
        "num_microbatches": 3,
        "pipeline_model": "optimistic",
    }


def _config(mode: str, **search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "example/model",
        "hardware_sku": "example_sku",
        "deployment_mode": [mode],
        "backend": ["vllm"],
        "gpu_budget": 16,
        "afd_pinned_topologies": [_topology()],
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"isl": 128, "osl": 32, "concurrency": 8, "num_request_ratio": 2},
    )


def _capabilities(mode: str) -> RunnerCapabilities:
    return RunnerCapabilities(supported_backend_topologies=(("vllm", mode),))


def test_pure_afd_uses_finite_generic_sampler_and_serializes_no_engine(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = _config(
        "afd",
        afd_pinned_topologies=[_topology(), _topology(n_a_nodes=2)],
    )

    (branch,) = enumerate_branches(config, runner_capabilities=_capabilities("afd"))
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert branch.flat_parallel_choices
    assert branch.domain_provenance["complete"] is True
    assert len(branch.parallel_configs) == 2
    assert isinstance(suggestion.parallel_config, AFDParallelConfig)
    assert suggestion.projection is not None
    assert sample["used_gpus"] == suggestion.parallel_config.total_gpus
    assert deployment.parallel_config["afd"] == sample["afd"]
    assert deployment.agg_engine_args is None
    assert deployment.prefill_engine_args is None
    assert deployment.decode_engine_args is None
    assert deployment.num_workers == 0
    assert deployment.performance_model_metadata["afd"]["provider"] == "unresolved"
    assert deployment.performance_model_metadata["afd"]["measurement_required"] is True


def test_afd_plus_pd_pairs_only_opposite_phase_companion(monkeypatch):
    companion = ReplicaParallelConfig(
        shape=ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=1),
        replicas=1,
    )
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    calls = []

    def fake_parallel_configs(*args, **kwargs):
        calls.append(kwargs)
        return [companion]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        fake_parallel_configs,
    )
    config = _config(
        "afd+pd",
        afd_phase="decode",
        afd_companion_parallel_configs=[{"tp": 4}],
        prefill_max_num_batched_tokens=[4096],
        prefill_max_num_seqs=[8],
    )

    (branch,) = enumerate_branches(config, runner_capabilities=_capabilities("afd+pd"))
    suggestion = RandomBranchSampler(branch, seed=3).suggest(1)[0]
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert calls[0]["deployment_mode"] == "agg"
    assert calls[0]["role_runtime"] == {"agg": (4096, 8, 0.9, None)}
    assert suggestion.parallel_config.companion_role == "prefill"
    assert sample["used_gpus"] == 12
    assert deployment.num_prefill_workers == 1
    assert deployment.prefill_engine_args["worker_type"] == "prefill"
    assert deployment.decode_engine_args is None
    assert "decode_tp" not in deployment.parallel_config


@pytest.mark.parametrize("mode", ["afd", "afd+pd"])
@pytest.mark.parametrize("pinned", [False, True])
def test_afd_runner_capability_gate_fails_closed(monkeypatch, mode, pinned):
    def unexpected_model_lookup(*args, **kwargs):
        pytest.fail("runner-incompatible AFD must be rejected before model lookup")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        unexpected_model_lookup,
    )

    with pytest.raises(RunnerIncompatibleError, match=r"runner-incompatible backends=\['vllm'\]"):
        enumerate_branches(
            _config(mode, afd_pinned_topologies=[_topology()] if pinned else []),
            runner_capabilities=RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),)),
        )


@pytest.mark.parametrize("pinned", [False, True])
def test_mixed_afd_runner_incompatibility_preserves_explicit_pin_scope(monkeypatch, pinned):
    def unexpected_model_lookup(*args, **kwargs):
        pytest.fail("runner-incompatible AFD must be rejected before model lookup")

    monkeypatch.setattr("aisimulate.sweeper.search_space.resolve_model_hardware", unexpected_model_lookup)
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [ReplicaParallelConfig(shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)],
    )
    config = _config(
        "afd",
        deployment_mode=["agg", "afd"],
        afd_pinned_topologies=[_topology()] if pinned else [],
    )
    if pinned:
        with pytest.raises(NoViableParallelConfig, match="runner-incompatible") as error:
            enumerate_branches(config, runner_capabilities=_capabilities("agg"))
        assert not isinstance(error.value, RunnerIncompatibleError)
    else:
        with pytest.warns(UserWarning, match="runner-incompatible"):
            (branch,) = enumerate_branches(config, runner_capabilities=_capabilities("agg"))
        assert branch.deployment_mode == "agg"


def test_mixed_afd_terminal_failure_defers_warning_and_preserves_runner_details(monkeypatch):
    monkeypatch.setattr("aisimulate.sweeper.search_space.parallel_configs_for", lambda *args, **kwargs: [])
    config = _config("afd", deployment_mode=["afd", "agg"], afd_pinned_topologies=[])

    with pytest.raises(NoViableParallelConfig, match="runner-incompatible") as error:
        enumerate_branches(config, runner_capabilities=_capabilities("agg"))

    assert not isinstance(error.value, RunnerIncompatibleError)


def test_afd_rejects_kv_relative_load_until_capacity_is_exposed(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = SmartSearchConfig(
        search_space=_config("afd").search_space.model_dump(),
        workload={
            "isl": 128,
            "osl": 32,
            "kv_load_ratio": 0.5,
            "num_request_ratio": 2,
        },
    )

    with pytest.raises(AFDInfeasible) as exc_info:
        enumerate_branches(config, runner_capabilities=_capabilities("afd"))

    assert exc_info.value.category is AFDReasonCategory.INVALID_TOPOLOGY
    assert "KV capacity" in exc_info.value.detail


def test_afd_complete_domain_limit_fails_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = _config(
        "afd",
        afd_pinned_topologies=[],
        afd_tp_a_candidates=[1],
        afd_batch_size_candidates=[1],
        afd_microbatch_candidates=[3],
        afd_pipeline_model_candidates=["serial"],
        afd_max_candidates=2,
    )

    with pytest.raises(AFDInfeasible) as exc_info:
        enumerate_branches(config, runner_capabilities=_capabilities("afd"))

    assert exc_info.value.category is AFDReasonCategory.CANDIDATE_LIMIT


def test_afd_combined_product_limit_fails_during_generation(monkeypatch):
    companions = [
        ReplicaParallelConfig(
            shape=ParallelShape(tp=tp, dp=1, moe_tp=1, moe_ep=1),
            replicas=1,
        )
        for tp in (1, 2, 4)
    ]
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: companions,
    )
    config = _config("afd+pd", afd_max_candidates=2)

    with pytest.raises(AFDInfeasible) as exc_info:
        enumerate_branches(config, runner_capabilities=_capabilities("afd+pd"))

    assert exc_info.value.category is AFDReasonCategory.CANDIDATE_LIMIT
    assert exc_info.value.provenance == {
        "generated_count": 3,
        "count_is_lower_bound": True,
    }


def test_sweeper_runs_afd_branch_through_an_explicitly_capable_runner(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    monkeypatch.setattr(
        "aisimulate.sweeper.search.resolve_backend_version",
        lambda *args, **kwargs: "test",
    )

    class Runner:
        def __init__(self):
            self.specs: list[ReplaySpec] = []

        def run(self, spec: ReplaySpec, *, output_requirements=None) -> ReplayReport:
            self.specs.append(spec)
            return ReplayReport(
                metrics={"output_throughput_tok_s": 10.0, "gpu_hours": 1.0}
            )

        def close(self) -> None:
            pass

    class Factory:
        def __init__(self):
            self.runner = Runner()

        def capabilities(self) -> RunnerCapabilities:
            return _capabilities("afd")

        def create(self, worker_id: int) -> Runner:
            return self.runner

    class PerformanceModel:
        def measure(self, request):
            phases = (
                ("prefill", "decode")
                if request.topology.phase.value == "both"
                else (request.topology.phase.value,)
            )
            return tuple(
                AFDLayerTimes(
                    phase=phase,
                    attention_ms=1.0,
                    ffn_ms=2.0,
                    a_to_f_ms=0.1,
                    f_to_a_ms=0.2,
                    num_layers=32,
                    provenance={"provider": "test"},
                )
                for phase in phases
            )

    factory = Factory()
    config = _config("afd")
    config.sweep.max_rounds = 1
    config.sweep.candidates_per_round = 1
    config.sweep.parallel_evals = 1
    config.sweep.algorithm = "random"

    result = Sweeper(
        runner_factory=factory,
        afd_performance_model=PerformanceModel(),
        show_progress=False,
    ).run(config, top_n=None)

    assert len(result.selected_candidates) == 1
    assert result.selected_candidates[0].used_gpus == 8
    assert factory.runner.specs[0].backend_deployment.deployment_mode == "afd"
    assert factory.runner.specs[0].backend_deployment.agg_engine_args is None
    assert (
        factory.runner.specs[0].backend_deployment.performance_model_metadata["afd"][
            "provider"
        ]
        == "test"
    )


def test_sweeper_runs_pure_both_phase_afd_through_engine_runner(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    monkeypatch.setattr(
        "aisimulate.sweeper.search.resolve_backend_version",
        lambda *args, **kwargs: "test",
    )

    class PerformanceModel:
        def measure(self, request):
            return tuple(
                AFDLayerTimes(
                    phase=phase,
                    attention_ms=1.0,
                    ffn_ms=1.0,
                    a_to_f_ms=0.0,
                    f_to_a_ms=0.0,
                    num_layers=1,
                    provenance={"provider": "test"},
                )
                for phase in ("prefill", "decode")
            )

    config = _config("afd", afd_phase="both")
    config.sweep.max_rounds = 1
    config.sweep.candidates_per_round = 1
    config.sweep.parallel_evals = 1
    config.sweep.algorithm = "random"

    result = Sweeper(
        runner_factory=EngineReplayRunnerFactory(),
        afd_performance_model=PerformanceModel(),
        show_progress=False,
    ).run(config, top_n=None)

    assert result.counts.feasible == 1
    assert result.selected_candidates[0].metrics["completed_requests"] == 16.0
    assert result.selected_candidates[0].metrics["output_throughput_tok_s"] > 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deployment_mode": ["afd"]}, "memory-qualified afd_batch_size_candidates"),
        (
            {
                "deployment_mode": ["afd+pd"],
                "afd_phase": "both",
                "afd_batch_size_candidates": [1],
            },
            "requires afd_phase prefill or decode",
        ),
    ],
)
def test_afd_search_contract_rejects_ambiguous_domains(overrides, message):
    with pytest.raises(ValidationError, match=message):
        SmartSearchConfig(
            search_space={
                "model_name": "example/model",
                "hardware_sku": "example_sku",
                **overrides,
            },
            workload={"trace_path": "/tmp/afd-trace.jsonl"},
        )
