# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit domains, capability boundaries and allocation refusal regressions."""

import pytest

import aisimulate.sweeper.model_hw as model_hw
from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.model_hw import ModelHardware
from aisimulate.sweeper.parallel_enum import (
    PreparationBudget,
    RoleParallelCandidates,
    SearchSpaceLimitError,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
)
from aisimulate.sweeper.parallel_projection import ParallelConfigProjector
from aisimulate.sweeper.search_space import _role_domain, enumerate_branches


def config(**overrides):
    return SmartSearchConfig(
        search_space={
            "model_name": "test/model",
            "hardware_sku": "h200_sxm",
            "gpu_budget": 32,
            "backend": ["sglang"],
            "deployment_mode": ["agg"],
            "context_length": 128,
            **overrides,
        },
        workload={"isl": 16, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
    )


@pytest.mark.parametrize(
    "backend,is_moe,gpus,expected",
    [
        ("vllm", False, 8, {(8, 1), (4, 2), (2, 4)}),
        ("sglang", True, 1, {(1, 1)}),
    ],
)
def test_explicit_pipeline_shapes_match_aic_enumeration(backend, is_moe, gpus, expected):
    from aiconfigurator_core.sdk.common import BackendName
    from aiconfigurator_core.sdk.utils import enumerate_parallel_config

    dims = dict(
        tp_candidates=(1, 2, 4, 8),
        pp_candidates=(1, 2, 4),
        attention_dp_candidates=(1,),
        moe_tp_candidates=(1,),
        moe_ep_candidates=(1,),
    )
    actual = enumerate_worker_shapes(is_moe=is_moe, backend=backend, gpus_per_worker=gpus, **dims)
    reference = enumerate_parallel_config(
        is_moe=is_moe,
        backend=BackendName(backend),
        num_gpu_list=[gpus],
        tp_list=[1, 2, 4, 8],
        pp_list=[1, 2, 4],
        dp_list=[1],
        moe_tp_list=[1],
        moe_ep_list=[1],
        cp_list=[1],
    )
    assert {(s.tp, s.pp) for s in actual} == expected
    assert {(s.tp, s.pp, s.dp, s.moe_tp, s.moe_ep, s.cp) for s in actual} == set(map(tuple, reference))


def test_preparation_refuses_before_worker_range_or_model_allocation(monkeypatch):
    monkeypatch.setattr(model_hw, "resolve_model_hardware", lambda *a, **k: pytest.fail("model lookup"))
    cfg = config(
        parallel_independent_by_mode={"agg": {"tp": [1]}},
        parallel_independent_log_ranges_by_mode={"agg": {"tp": [1, 10**9]}},
        max_parallel_combinations=100,
    )
    with pytest.raises(SearchSpaceLimitError, match="integer_domain"):
        enumerate_branches(cfg)
    with pytest.raises(SearchSpaceLimitError, match="workers"):
        enumerate_parallel_configs(
            is_moe=False,
            backend="vllm",
            gpu_budget=10**9,
            gpus_per_worker_candidates=(1,),
            preparation=PreparationBudget(10000, 100),
        )


def test_retained_domain_ceiling_raises_instead_of_truncating():
    with pytest.raises(SearchSpaceLimitError, match="max_parallel_configs=2"):
        enumerate_parallel_configs(
            is_moe=False,
            backend="vllm",
            gpu_budget=4,
            gpus_per_worker_candidates=(1,),
            preparation=PreparationBudget(max_configs=2),
        )


def test_explicit_cp_choices_survive_default_resolution():
    cfg = config(
        agg_num_gpu_candidates=[1, 16], agg_tp_candidates=[1], agg_dp_candidates=[1], agg_cp_candidates=[1, 16]
    )
    domain = _role_domain(cfg.search_space, "agg", "agg", PreparationBudget())
    assert domain.cp == (1, 16)
    shapes = enumerate_worker_shapes(
        is_moe=False, backend="sglang", gpus_per_worker=16, tp_candidates=domain.tp, cp_candidates=domain.cp
    )
    assert [s.cp for s in shapes] == [16]


def test_worker_only_override_preserves_default_gpu_ladder():
    domain = _role_domain(config(agg_num_workers_candidates=[1]).search_space, "agg", "agg", PreparationBudget())
    assert domain.gpus_per_worker == RoleParallelCandidates().gpus_per_worker


def test_asymmetric_worker_limits_and_pruning_are_deterministic():
    domain = RoleParallelCandidates(
        gpus_per_worker=(1,), tp=(1,), pp=(1,), attention_dp=(1,), moe_tp=(1,), moe_ep=(1,), cp=(1,), workers=(1, 2, 3)
    )
    reports = []
    for _ in range(2):
        budget = PreparationBudget()
        candidates = enumerate_disagg_configs(
            is_moe=False,
            backend="sglang",
            gpu_budget=6,
            prefill_candidates=domain,
            decode_candidates=domain,
            max_prefill_workers=1,
            max_decode_workers=2,
            max_gpu_per_replica=3,
            preparation=budget,
        )
        assert {(c.prefill.replicas, c.decode.replicas) for c in candidates} == {(1, 1), (1, 2)}
        reports.append(budget.as_dict())
    assert reports[0] == reports[1]
    assert reports[0]["stages"]["topology.prefill.workers"]["worker_or_gpu_limit"] == 2


def test_expanded_sdk_pool_preserves_equal_gpu_pipeline_identity(monkeypatch):
    facts = ModelHardware("test/model", "h200_sxm", "sglang", False, False, False, 1, 1, 8, 128)
    monkeypatch.setattr(model_hw, "resolve_model_hardware", lambda *a, **k: facts)
    monkeypatch.setattr(model_hw, "feasible_shape_tokens", lambda shapes, **k: dict.fromkeys(shapes, 1024))
    cfg = config(
        agg_num_gpu_candidates=[8],
        agg_tp_candidates=[4, 8],
        agg_pp_candidates=[1, 2],
        agg_dp_candidates=[1],
        agg_num_workers_candidates=[1],
    )
    branch = enumerate_branches(cfg)[0]
    assert {(c.shape.tp, c.shape.pp) for c in branch.parallel_configs} == {(4, 2), (8, 1)}
    projector = ParallelConfigProjector(branch)
    assert branch.flat_parallel_choices
    assert {projector.project({"parallel_config_choice": i}, "sglang").config for i in range(2)} == set(
        branch.parallel_configs
    )


def test_public_context_and_pipeline_lower_to_expanded_domain():
    public = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "context_length": 128,
                "model": "test/model",
                "hardware": "h200_sxm",
                "backend": "sglang",
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "preset": False,
                            "tensor": 1,
                            "attention_data": 1,
                            "context": {"choices": [1, 16]},
                            "pipeline": {"choices": [1, 2]},
                        }
                    }
                },
            },
            "optimization": {},
        }
    )
    lowered = recommendation_to_sweeper(public)
    domain = _role_domain(lowered.search_space, "agg", "agg", PreparationBudget())
    assert domain.cp == (1, 16)
    assert domain.pp == (1, 2)


def test_budget_report_is_json_compatible():
    import json

    budget = PreparationBudget()
    enumerate_worker_shapes(is_moe=False, backend="vllm", gpus_per_worker=1, preparation=budget)
    assert json.loads(json.dumps(budget.as_dict())) == budget.as_dict()


def test_mixed_custom_role_preserves_pipeline_identity():
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.search_space import BranchSpace

    prefill = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    candidates = tuple(
        DisaggParallelConfig(prefill, ReplicaParallelConfig(ParallelShape(tp=tp, pp=pp, dp=1, moe_tp=1, moe_ep=1), 1))
        for tp, pp in ((8, 1), (4, 2))
    )
    branch = BranchSpace(
        deployment_mode="disagg",
        parallel_configs=candidates,
        supported_backends=dict.fromkeys(candidates, frozenset({"sglang"})),
        knob_choices={"backend": ["sglang"]},
        gpu_budget=9,
        parallel_custom_choices={"prefill": (prefill,)},
    )
    projector = ParallelConfigProjector(branch)
    assert {projector.project({"decode_pipeline_stages": pp}, "sglang").config for pp in (1, 2)} == set(candidates)


def test_older_runner_cannot_accept_pipeline_or_context():
    from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.replay import RunnerCapabilities
    from aisimulate.sweeper.search_space import _runner_supports_parallel_config

    legacy = RunnerCapabilities()
    for dims in ({"pp": 2}, {"cp": 2}):
        candidate = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1, **dims), 1)
        assert not _runner_supports_parallel_config(legacy, "agg", candidate)


@pytest.mark.parametrize(
    "overrides",
    [
        {"agg_pp_candidates": [2], "parallel_configs": [{"tp": 1}]},
        {"prefill_pp_candidates": [2]},
        {"deployment_mode": ["afd"], "agg_cp_candidates": [2]},
    ],
)
def test_conflicting_or_inactive_explicit_domains_are_rejected(overrides):
    with pytest.raises(ValueError, match="require|cannot be combined"):
        config(**overrides)


@pytest.mark.parametrize("field", ["tp", "pp", "attention_dp", "moe_tp", "moe_ep", "cp", "replicas"])
@pytest.mark.parametrize("value", [-1, 0, True, 1.5, "2"])
@pytest.mark.parametrize("preset_form", ["legacy", "aggregated", "disaggregated", "custom"])
def test_sdk_presets_reject_invalid_dimensions_before_enumeration(field, value, preset_form):
    shape = {"tp": 1, field: value}
    if preset_form == "legacy":
        overrides = {"parallel_configs": [shape]}
    elif preset_form == "aggregated":
        overrides = {"parallel_configs_by_mode": {"agg": [shape]}}
    elif preset_form == "disaggregated":
        overrides = {
            "deployment_mode": ["disagg"],
            "parallel_configs_by_mode": {"disagg": [{"prefill": {"tp": 1}, "decode": shape}]},
        }
    else:
        overrides = {
            "deployment_mode": ["disagg"],
            "parallel_custom_configs_by_mode": {"disagg": {"prefill": [shape]}},
        }
    with pytest.raises(ValueError, match=rf"parallel_configs\.{field} must be a positive integer"):
        config(**overrides)


@pytest.mark.parametrize("field", ["max_prefill_workers", "max_decode_workers"])
def test_inactive_worker_limit_is_rejected(field):
    with pytest.raises(ValueError, match="worker limits require"):
        config(**{field: 1})


def test_shared_and_role_gpu_domains_cannot_silently_override_each_other():
    with pytest.raises(ValueError, match="shared gpus_per_worker_candidates"):
        enumerate_disagg_configs(
            is_moe=False,
            backend="sglang",
            gpu_budget=8,
            gpus_per_worker_candidates=(1,),
            prefill_candidates=RoleParallelCandidates(gpus_per_worker=(2,)),
        )


@pytest.mark.parametrize("cp", [0, 2, True, 1.5])
def test_afd_companion_cp_is_rejected_before_model_lookup(cp):
    with pytest.raises(ValueError, match="AFD companion context parallelism must be 1"):
        config(
            deployment_mode=["afd+pd"],
            afd_phase="prefill",
            afd_batch_size_candidates=[1],
            afd_companion_parallel_configs=[{"tp": 1, "cp": cp}],
        )


def _public_with_expanded_ranges():
    return CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "test/model",
                "hardware": "h200_sxm",
                "backend": "sglang",
                "context_length": 128,
                "workers": {
                    "aggregated": {
                        "scheduler": {"max_sequences": {"range": {"min": 1, "max": 4, "step": 1}}},
                        "parallelism": {
                            "preset": False,
                            "context": {"range": {"min": 1, "max": 4, "step": 1}},
                        },
                    }
                },
            },
            "optimization": {},
        }
    )


def test_public_ranges_share_one_preparation_limit(monkeypatch):
    import aisimulate.recommend as recommend

    monkeypatch.setattr(recommend, "PreparationBudget", lambda: PreparationBudget(max_combinations=7))
    with pytest.raises(SearchSpaceLimitError, match="input.integer_range"):
        recommendation_to_sweeper(_public_with_expanded_ranges())


def test_public_lowering_diagnostics_seed_each_fresh_enumeration_budget():
    smart = recommendation_to_sweeper(_public_with_expanded_ranges())
    first = smart.search_space.new_preparation_budget()
    assert first.considered == 8
    assert first.stages["input.integer_range"]["considered"] == 8
    first.reserve(1, "subsequent_work")
    assert smart.search_space.new_preparation_budget().considered == 8
    smart.search_space.max_parallel_combinations = 7
    with pytest.raises(SearchSpaceLimitError, match="input.integer_range"):
        enumerate_branches(smart)


def test_public_cp_capability_uses_model_resolution_and_errors(monkeypatch):
    import aiconfigurator_core.sdk.models as models

    assert models.supports_context_parallelism("Qwen/Qwen3-32B", "sglang")
    assert not models.supports_context_parallelism("Qwen/Qwen3-32B", "vllm")
    monkeypatch.setattr(models, "get_model_family", lambda _: "unregistered-test-family")
    with pytest.raises(ValueError, match="Unknown model family"):
        models.supports_context_parallelism("test/model", "sglang")


def test_float_range_reserves_accumulation_endpoint_before_allocating():
    from aisimulate.recommend import _choices

    domain = {"range": {"min": 0.0, "max": 1.0, "step": 0.1}}
    with pytest.raises(SearchSpaceLimitError, match="input.integer_range"):
        _choices(domain, default=[], preparation=PreparationBudget(max_combinations=10))
    budget = PreparationBudget(max_combinations=11)
    assert len(_choices(domain, default=[], preparation=budget)) == budget.considered == 11


def test_float_range_rejects_nonadvancing_step():
    from aisimulate.recommend import _choices

    with pytest.raises(ValueError, match="step does not advance"):
        _choices({"range": {"min": 1e16, "max": 1e16 + 2, "step": 1.0}}, default=[])
    assert _choices({"range": {"min": 1e16, "max": 1e16, "step": 1.0}}, default=[]) == [1e16]


@pytest.mark.parametrize("bounds", [(3, 1, 1), (1, 3, -1), (1, 3, 0)])
def test_invalid_linear_range_does_not_change_preparation_budget(bounds):
    from aisimulate.recommend import _choices

    budget = PreparationBudget(max_combinations=5)
    budget.reserve(4, "earlier")
    before = budget.as_dict()
    minimum, maximum, step = bounds
    with pytest.raises(ValueError, match="linear range requires"):
        _choices({"range": {"min": minimum, "max": maximum, "step": step}}, default=[], preparation=budget)
    assert budget.as_dict() == before
    with pytest.raises(SearchSpaceLimitError):
        budget.reserve(2, "later")


def test_preparation_reservation_cannot_refund_prior_work():
    budget = PreparationBudget(max_combinations=5)
    budget.reserve(4, "earlier")
    before = budget.as_dict()
    with pytest.raises(ValueError, match="reservation must be non-negative"):
        budget.reserve(-1, "invalid")
    assert budget.as_dict() == before
    with pytest.raises(SearchSpaceLimitError):
        budget.reserve(2, "later")
