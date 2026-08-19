# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic rapid/thorough integration for pure and combined AFD search."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import aisimulate.sweeper.search as search_mod
import aisimulate.sweeper.search_space as search_space_mod
from aisimulate.sweeper.afd import AFDParallelConfig
from aisimulate.sweeper.config import SearchPolicy, SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.model_hw import ModelHardware, NoViableParallelConfig
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.parallel_projection import ParallelConfigProjector
from aisimulate.sweeper.replay import ReplayReport, RunnerCapabilities
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import ExhaustiveBranchSampler, Suggestion
from aisimulate.sweeper.search import Sweeper
from aisimulate.sweeper.search_space import enumerate_branches

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")


def _facts(*, backend: str = "vllm", is_moe: bool = False) -> ModelHardware:
    return ModelHardware(
        model_name="model",
        hardware_sku="test",
        backend=backend,
        is_moe=is_moe,
        mla=is_moe,
        enable_wideep=False,
        weight_bytes=1,
        vram_per_gpu=80,
        gpus_per_node=4,
        max_context=8192,
        model_family="MODEL",
        num_experts=8 if is_moe else 0,
        default_gpus_per_worker=(1, 2, 4),
        default_pp_candidates=(1,),
        default_cp_candidates=(1,),
    )


def _config(mode: str, **overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "model",
        "hardware_sku": "test",
        "backend": ["vllm"],
        "deployment_mode": [mode],
        "gpu_budget": 16,
        "afd_tp_a_candidates": [1],
        "afd_batch_size_candidates": [8],
        "afd_microbatch_candidates": [2],
        "afd_pipeline_model_candidates": ["serial"],
    }
    search_space.update(overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"trace_path": TRACE},
        goal={"target": "throughput"},
        sweep={"max_rounds": 1, "parallel_evals": 1, "candidates_per_round": 2},
    )


def _capabilities(mode: str) -> RunnerCapabilities:
    return RunnerCapabilities(supported_backend_topologies=(("vllm", mode),))


def _patch_facts(monkeypatch, *, is_moe: bool = False) -> None:
    monkeypatch.setattr(
        search_space_mod,
        "resolve_model_hardware",
        lambda model, hardware, *, backend: _facts(
            backend=backend, is_moe=is_moe
        ),
    )


def test_afd_configuration_rejects_ambiguous_pins_and_combined_both_phase():
    with pytest.raises(ValidationError, match="exactly one AFD deployment_mode"):
        _config(
            "afd",
            deployment_mode=["afd", "afd+pd"],
            afd_pinned_topologies=[
                {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8}
            ],
        )

    with pytest.raises(ValidationError, match="requires afd_phase prefill or decode"):
        _config("afd+pd", afd_phase="both")


def test_thorough_pinned_pure_afd_materializes_lossless_topology(monkeypatch):
    _patch_facts(monkeypatch)
    config = _config(
        "afd",
        afd_phase="both",
        afd_pinned_topologies=[
            {
                "n_a_nodes": 1,
                "n_f_nodes": 1,
                "tp_a": 1,
                "a_batch_size": 8,
                "num_microbatches": 2,
                "pipeline_model": "serial",
            }
        ],
    )
    (branch,) = enumerate_branches(
        config, runner_capabilities=_capabilities("afd")
    )

    sampler = ExhaustiveBranchSampler(branch)
    (suggestion,) = sampler.suggest(10)
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert sampler.candidate_count == 1
    assert branch.domain_provenance["afd_enumeration"]["domain"] == "pinned"
    assert sample["used_gpus"] == 8
    assert sample["afd_attention_gpus"] == 4
    assert sample["afd_ffn_gpus"] == 4
    assert deployment.deployment_mode == "afd"
    assert deployment.agg_engine_args is None
    assert deployment.prefill_engine_args is None
    assert deployment.decode_engine_args is None
    assert deployment.parallel_config["afd"]["phase"] == "both"
    assert (
        deployment.parallel_config["afd_provenance"]["gpu_accounting"][
            "total_gpus"
        ]
        == 8
    )
    _capabilities("afd").require_compatible(
        search_mod.ReplaySpec(
            backend_deployment=deployment,
            workload={},
            goal={},
        )
    )


def test_rapid_searched_pure_afd_uses_projectable_complete_legal_pool(monkeypatch):
    _patch_facts(monkeypatch)
    config = _config("afd", afd_phase="decode")
    (branch,) = enumerate_branches(
        config, runner_capabilities=_capabilities("afd")
    )

    projector = ParallelConfigProjector(branch)
    projection = projector.project({}, "vllm")

    assert len(branch.parallel_configs) == 6
    assert branch.domain_provenance["afd_enumeration"] == {
        "schema_version": 1,
        "source": "aiconfigurator.sdk.task_v2.build_afd_parallel_lists",
        "domain": "searched",
        "complete": True,
        "candidate_order": [
            "n_a_nodes",
            "n_f_nodes",
            "tp_a",
            "a_batch_size",
            "f_moe_ep_size",
            "num_microbatches",
            "pipeline_model",
        ],
        "generated_count": 6,
        "accepted_topologies": 6,
        "rejection_counts": {
            "gpu_budget": 0,
            "af_ratio": 0,
            "expert_divisibility": 0,
            "duplicate_pipeline_regime": 0,
        },
        "truncated": False,
    }
    assert isinstance(projection.config, AFDParallelConfig)
    assert projection.config in branch.parallel_configs
    assert projection.actual_features["afd_pipeline_model"] == "serial"


def test_combined_afd_uses_role_domain_and_accounts_for_every_gpu(monkeypatch):
    _patch_facts(monkeypatch)
    seen = {}
    companions = [
        ReplicaParallelConfig(
            ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
        ),
        ReplicaParallelConfig(
            ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=3
        ),
        ReplicaParallelConfig(
            ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=40
        ),
    ]

    def fake_parallel_configs(*args, **kwargs):
        seen.update(kwargs)
        return companions

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", fake_parallel_configs)
    config = _config(
        "afd+pd",
        gpu_budget=12,
        afd_phase="decode",
        prefill_num_gpu_candidates=[1],
        prefill_tp_candidates=[1],
        prefill_num_workers_candidates=[1, 3, 40],
        prefill_batch_size_candidates=[1],
        prefill_context_tokens_candidates=[8192],
    )
    (branch,) = enumerate_branches(
        config, runner_capabilities=_capabilities("afd+pd")
    )

    assert seen["deployment_mode"] == "agg"
    assert seen["agg_candidates"].workers == (1, 3, 40)
    assert len(branch.parallel_configs) == 2
    assert branch.domain_provenance["combined_pruning"] == {
        "companion_worker_ceiling": 3,
        "gpu_budget": 4,
        "minimum_gpu_budget": 0,
        "runner_attention_dp": 0,
    }
    assert ExhaustiveBranchSampler(branch).candidate_count == 2

    selected = max(branch.parallel_configs, key=lambda item: item.total_gpus)
    sample = unroll_sample(
        search_space=config.search_space,
        selection={
            "deployment_mode": "afd+pd",
            "backend": "vllm",
            "prefill_batch_size": 1,
            "prefill_context_tokens": 8192,
        },
        parallel_config=selected,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert sample["afd_attention_gpus"] == 4
    assert sample["afd_ffn_gpus"] == 4
    assert sample["afd_companion_gpus"] == 3
    assert sample["used_gpus"] == 11
    assert deployment.num_prefill_workers == 3
    assert deployment.prefill_engine_args["max_num_seqs"] == 1
    assert deployment.decode_engine_args is None
    assert deployment.parallel_config["afd_provenance"]["gpu_accounting"] == {
        "attention_gpus": 4,
        "ffn_gpus": 4,
        "companion_gpus": 3,
        "total_gpus": 11,
    }


def test_prefill_afd_forces_decode_companion_cp_and_materializes_decode(monkeypatch):
    _patch_facts(monkeypatch)
    seen = {}
    companion = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
    )

    def fake_parallel_configs(*args, **kwargs):
        seen.update(kwargs)
        return [companion]

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", fake_parallel_configs)
    config = _config(
        "afd+pd",
        gpu_budget=12,
        afd_phase="prefill",
        decode_batch_size_candidates=[16],
        decode_context_tokens_candidates=[8192],
    )
    (branch,) = enumerate_branches(
        config, runner_capabilities=_capabilities("afd+pd")
    )
    assert seen["agg_candidates"].cp == (1,)

    selected = min(branch.parallel_configs, key=lambda item: item.total_gpus)
    sample = unroll_sample(
        search_space=config.search_space,
        selection={
            "deployment_mode": "afd+pd",
            "backend": "vllm",
            "decode_batch_size": 16,
            "decode_context_tokens": 8192,
        },
        parallel_config=selected,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert selected.companion_role == "decode"
    assert deployment.decode_engine_args["aic_cp_size"] == 1
    assert deployment.num_decode_workers == 1
    assert deployment.prefill_engine_args is None


def test_runner_capability_gate_fails_closed_before_afd_materialization(monkeypatch):
    _patch_facts(monkeypatch)
    config = _config("afd")

    with pytest.warns(UserWarning, match="no configured backend.*AFD"):
        with pytest.raises(NoViableParallelConfig):
            enumerate_branches(
                config,
                runner_capabilities=RunnerCapabilities(
                    supported_backend_topologies=(("vllm", "agg"),)
                ),
            )


class _AFDRunner:
    def __init__(self):
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        assert spec.backend_deployment.parallel_config["afd"]
        return ReplayReport(
            metrics={"output_throughput_tok_s": 100.0, "gpu_hours": 1.0}
        )

    def close(self):
        return None


class _AFDRunnerFactory:
    def __init__(self):
        self.runner = _AFDRunner()

    def capabilities(self):
        return _capabilities("afd")

    def create(self, worker_id):
        return self.runner


class _FiniteRapidSampler:
    def __init__(self, branch, study_id, objectives=None, seed=None):
        self.branch = branch
        self.index = 0

    def suggest(self, count):
        end = min(self.index + count, len(self.branch.parallel_configs))
        suggestions = [
            Suggestion(
                selection={"deployment_mode": "afd", "backend": "vllm"},
                parallel_config=self.branch.parallel_configs[index],
                handle=index,
            )
            for index in range(self.index, end)
        ]
        self.index = end
        return suggestions

    def observe(self, suggestion, metrics):
        return None

    def observe_infeasible(self, suggestion, reason):
        return None


@pytest.mark.parametrize(
    ("policy", "pinned", "sampler_factory", "expected", "complete"),
    [
        ("rapid", False, _FiniteRapidSampler, 2, False),
        ("thorough", True, None, 1, True),
    ],
)
def test_generic_sweeper_runs_rapid_searched_and_thorough_pinned_afd(
    monkeypatch, policy, pinned, sampler_factory, expected, complete
):
    _patch_facts(monkeypatch)
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "test")
    overrides = {"afd_phase": "decode"}
    if pinned:
        overrides["afd_pinned_topologies"] = [
            {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8}
        ]
    config = _config("afd", **overrides)
    config.sweep.policy = SearchPolicy(policy)
    factory = _AFDRunnerFactory()
    kwargs = {
        "runner_factory": factory,
        "show_progress": False,
    }
    if sampler_factory is not None:
        kwargs["sampler_factory"] = sampler_factory
    sweeper = Sweeper(**kwargs)

    candidates = sweeper.run(config)

    assert len(candidates) == expected
    assert all(candidate.config["deployment_mode"] == "afd" for candidate in candidates)
    assert all(candidate.used_gpus >= 8 for candidate in candidates)
    assert sweeper.last_report.complete is complete
    assert sweeper.last_report.approximate is (not complete)
    assert sweeper.last_report.finite_candidate_count == (1 if complete else None)
