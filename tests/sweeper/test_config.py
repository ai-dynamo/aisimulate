# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-independent Sweeper configuration contracts."""

import pytest
from pydantic import ValidationError

from aisimulate.config import CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper import OptimizationTarget, SmartSearchConfig
from aisimulate.sweeper.config import (
    OptimizationGoal,
    SearchSpace,
    SLATarget,
    SweepConfig,
    Workload,
)


def _search_space(**overrides):
    return {"model_name": "m", "hardware_sku": "h200_sxm", **overrides}


def _workload(**overrides):
    return {
        "isl": 4000,
        "osl": 1000,
        "concurrency": 2,
        "num_request_ratio": 10,
        **overrides,
    }


def test_min_gpus_requires_sla_and_rejects_pareto_use():
    with pytest.raises(ValidationError, match="SLA bound"):
        OptimizationGoal(target="min_gpus")
    with pytest.raises(ValidationError, match="not a Pareto objective"):
        OptimizationGoal(target="pareto", pareto_objectives=["min_gpus", "throughput"])
    with pytest.raises(ValidationError, match="only supported with min_gpus"):
        OptimizationGoal(target="throughput", min_goodput_rps=2)


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf"), True])
def test_min_gpus_rejects_invalid_min_goodput(minimum):
    with pytest.raises(ValidationError):
        OptimizationGoal(target="min_gpus", sla={"itl_ms": 30}, min_goodput_rps=minimum)


@pytest.mark.parametrize(
    ("workload", "minimum", "error"),
    [
        (_workload(concurrency=None, request_rate=10), None, "requires min_goodput_rps"),
        (_workload(concurrency=None, request_rate=10), 11, "cannot exceed"),
        (_workload(concurrency=None, kv_load_ratio=0.5), None, "fixed synthetic"),
        ({"trace_path": "trace.jsonl"}, 1, "fixed synthetic"),
        (_workload(load_search_field="concurrency", load_choices=[2, 4]), None, "fixed synthetic"),
    ],
)
def test_min_gpus_sdk_validates_workload(workload, minimum, error):
    with pytest.raises(ValidationError, match=error):
        SmartSearchConfig(
            search_space=_search_space(),
            workload=workload,
            goal={"target": "min_gpus", "sla": {"itl_ms": 30}, "min_goodput_rps": minimum},
        )


def test_min_gpus_sdk_accepts_goodput_floor_equal_to_offered_request_rate():
    config = SmartSearchConfig(
        search_space=_search_space(),
        workload=_workload(concurrency=None, request_rate=10),
        goal={"target": "min_gpus", "sla": {"itl_ms": 30}, "min_goodput_rps": 10},
    )
    assert config.goal.min_goodput_rps == config.workload.request_rate == 10


def test_backend_only_yaml_and_adapter_search_space_load(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
search_space:
  deployment_mode: [agg]
  backend: [vllm]
  model_name: example/model
  hardware_sku: h200_sxm
  gpu_budget: 8
adapters:
  dynamo.planner:
    search_space:
      scaling_interval: [5, 10]
      load_predictor: [default, conservative]
workload:
  isl: 128
  osl: 16
  request_rate: 2
  num_request_ratio: 3
sweep:
  max_rounds: 2
  candidates_per_round: 1
"""
    )

    config = SmartSearchConfig.from_yaml(path)

    assert config.search_space.deployment_mode == ["agg"]
    assert config.search_space.backend == ["vllm"]
    assert config.adapters["dynamo.planner"].search_space == {
        "scaling_interval": [5, 10],
        "load_predictor": ["default", "conservative"],
    }
    assert config.workload.request_rate == 2
    assert config.sweep.max_rounds == 2


def test_aic_strict_sla_migration_yaml_loads(tmp_path):
    """Keep the command-migration guide's strict-SLA example executable."""
    path = tmp_path / "recommendation.yaml"
    path.write_text(
        """
traffic:
  source:
    type: synthetic
    input_tokens: 1024
    output_tokens: 128
  load:
    type: constant_rate
    requests_per_second: 4
  stop:
    requests_per_load_unit: 10
engine:
  mode: aggregated
  model: meta-llama/Meta-Llama-3.1-8B
  hardware: gb200
  backend: trtllm
  workers:
    aggregated: {}
evaluation:
  sla:
    ttft_ms: 800
    itl_ms: 30
optimization:
  target: throughput
  strict_sla: true
  constraints:
    max_candidate_gpus: 8
"""
    )

    public = CoreRecommendationConfig.from_yaml(path)
    config = recommendation_to_sweeper(public)

    assert config.goal.strict_sla
    assert config.goal.sla == SLATarget(ttft_ms=800, itl_ms=30)


def test_defaults_are_backend_only():
    config = SmartSearchConfig(
        search_space=_search_space(),
        workload=_workload(),
    )

    assert config.search_space.gpu_budget == 32
    assert config.search_space.prefill_block_size == 64
    assert config.search_space.prefill_max_num_seqs == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    assert config.adapters == {}
    assert config.goal.target is OptimizationTarget.THROUGHPUT
    assert config.sweep.parallel_evals == 16
    dumped = config.search_space.model_dump()
    assert not {"planner_scaling_policy", "router_mode", "num_g2_blocks"} & dumped.keys()


@pytest.mark.parametrize(
    ("overrides", "prefill", "decode"),
    [
        ({}, "h200_sxm", "h200_sxm"),
        ({"prefill_hardware_sku": "gb200"}, "gb200", "h200_sxm"),
        ({"decode_hardware_sku": "gb200"}, "h200_sxm", "gb200"),
        (
            {"prefill_hardware_sku": "h100_sxm", "decode_hardware_sku": "gb200"},
            "h100_sxm",
            "gb200",
        ),
    ],
)
def test_disagg_role_hardware_inherits_shared_sku(overrides, prefill, decode):
    search_space = SearchSpace(**_search_space(deployment_mode=["disagg"], **overrides))

    assert search_space.hardware_sku_for("prefill") == prefill
    assert search_space.hardware_sku_for("decode") == decode
    assert search_space.hardware_sku_for("agg") == "h200_sxm"


def test_role_hardware_requires_disagg_mode():
    with pytest.raises(ValidationError, match="require deployment_mode to include 'disagg'"):
        SearchSpace(
            **_search_space(
                deployment_mode=["agg"],
                prefill_hardware_sku="gb200",
            )
        )


def test_extra_fields_are_forbidden_at_each_boundary():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SmartSearchConfig(
            search_space=_search_space(bogus=1),
            workload=_workload(),
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SmartSearchConfig(
            search_space=_search_space(),
            workload=_workload(),
            adapters={"custom": {"search_space": {}, "bogus": 1}},
        )


def test_backend_choice_subset_is_accepted():
    config = SmartSearchConfig(
        search_space=_search_space(deployment_mode=["agg"], backend=["vllm", "sglang"]),
        workload=_workload(),
    )

    assert config.search_space.deployment_mode == ["agg"]
    assert config.search_space.backend == ["vllm", "sglang"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", ["bogus"]),
        ("deployment_mode", ["bogus"]),
        ("prefill_max_num_seqs", [0]),
        ("agg_max_num_batched_tokens", [-1]),
        ("backend", []),
    ],
)
def test_invalid_or_empty_backend_choices_are_rejected(field, value):
    with pytest.raises(ValidationError):
        SearchSpace(**_search_space(**{field: value}))


def test_pinned_parallel_config_requires_one_deployment_mode():
    with pytest.raises(ValidationError, match="exactly one mode"):
        SearchSpace(
            **_search_space(
                deployment_mode=["agg", "disagg"],
                parallel_configs=[{"tp": 4}],
            )
        )


@pytest.mark.parametrize(
    ("mode", "entry", "message"),
    [
        ("agg", {"replicas": 2}, "'tp' field"),
        ("disagg", {"tp": 4}, "prefill"),
        (
            "disagg",
            {"prefill": 1, "decode": {"tp": 1}},
            "prefill.*dict",
        ),
        (
            "disagg",
            {"prefill": {"tp": 1}, "decode": {"replicas": 1}},
            "decode.*'tp' field",
        ),
    ],
)
def test_parallel_config_shape_matches_mode(mode, entry, message):
    with pytest.raises(ValidationError, match=message):
        SearchSpace(
            **_search_space(
                deployment_mode=[mode],
                parallel_configs=[entry],
            )
        )


def test_well_formed_agg_and_disagg_parallel_configs_are_accepted():
    SearchSpace(
        **_search_space(
            deployment_mode=["agg"],
            parallel_configs=[{"tp": 4, "moe_ep": 4, "replicas": 2}],
        )
    )
    SearchSpace(
        **_search_space(
            deployment_mode=["disagg"],
            parallel_configs=[
                {
                    "prefill": {"tp": 8, "moe_ep": 8},
                    "decode": {"tp": 1, "attention_dp": 8, "moe_ep": 8},
                }
            ],
        )
    )


def test_trace_and_synthetic_workloads_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="must not set synthetic fields"):
        Workload(
            trace_path="/tmp/trace.jsonl",
            isl=1,
            osl=1,
            concurrency=1,
            num_request_ratio=1,
        )
    with pytest.raises(ValidationError, match="exactly one"):
        Workload(isl=1, osl=1, concurrency=1, request_rate=1, num_request_ratio=1)


def test_trace_closed_loop_cap_and_synthetic_helpers():
    trace = Workload(trace_path="/tmp/trace.jsonl", replay_concurrency=8)
    concurrency = Workload(isl=4000, osl=1000, concurrency=256, num_request_ratio=10)
    rate = Workload(isl=4000, osl=1000, request_rate=25, num_request_ratio=4)

    assert trace.is_trace_based
    assert trace.effective_in_flight_cap() == 8
    assert concurrency.effective_in_flight_cap() == 256
    assert concurrency.effective_in_flight_cap(8) == 8
    assert concurrency.resolved_request_count() == 2560
    assert concurrency.resolved_request_count(8) == 80
    assert concurrency.synthetic_arrival_interval_ms is None
    assert rate.synthetic_arrival_interval_ms == 40
    assert rate.resolved_request_count() == 100


def test_synthetic_random_length_options_are_validated():
    workload = Workload(
        **_workload(random_range_ratio=0.8, random_seed=7),
    )

    assert workload.random_range_ratio == 0.8
    assert workload.random_seed == 7

    for ratio in (0.0, -0.1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValidationError, match="random_range_ratio"):
            Workload(**_workload(random_range_ratio=ratio))
    for ratio in (True, "0.8"):
        with pytest.raises(ValidationError, match="random_range_ratio"):
            Workload(**_workload(random_range_ratio=ratio))
    for seed in (-1, 2**64):
        with pytest.raises(ValidationError, match="random_seed"):
            Workload(**_workload(random_seed=seed))
    for seed in (True, 1.0, "7"):
        with pytest.raises(ValidationError, match="random_seed"):
            Workload(**_workload(random_seed=seed))
    for lengths in ({"isl": 1}, {"osl": 1}):
        with pytest.raises(ValidationError, match="zero-token lower bound"):
            Workload(**_workload(**lengths, random_range_ratio=0.8))
    with pytest.raises(ValidationError, match="single-turn"):
        Workload(**_workload(random_range_ratio=0.8, turns_per_session=2))
    with pytest.raises(ValidationError, match="must not set synthetic fields"):
        Workload(trace_path="/tmp/trace.jsonl", random_range_ratio=0.8)


@pytest.mark.parametrize(
    "workload",
    [
        {"isl": 1, "osl": 1, "concurrency": 0, "num_request_ratio": 1},
        {"isl": 1, "osl": 1, "concurrency": 1.9, "num_request_ratio": 1},
        {"isl": 1, "osl": 1, "concurrency": 1},
        {"trace_path": "/tmp/trace.jsonl", "replay_concurrency": 0},
    ],
)
def test_invalid_workloads_are_rejected(workload):
    with pytest.raises(ValidationError):
        Workload(**workload)


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 2000.0), ("itl_ms", 30.0), ("e2e_ms", 5000.0)])
@pytest.mark.parametrize("target", [OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU])
def test_goodput_requires_at_least_one_sla_bound(target: OptimizationTarget, field: str, bound: float):
    with pytest.raises(ValidationError, match="require at least one SLA"):
        OptimizationGoal(target=target)

    goal = OptimizationGoal(
        target=target,
        sla=SLATarget(**{field: bound}),
    )
    assert getattr(goal.sla, field) == bound


def test_strict_sla_requires_a_bound_but_does_not_change_request_sla_shape():
    with pytest.raises(ValidationError, match="strict_sla requires"):
        OptimizationGoal(strict_sla=True)

    non_strict = OptimizationGoal(sla=SLATarget(itl_ms=30))
    assert not non_strict.strict_sla

    goal = OptimizationGoal(
        target=OptimizationTarget.THROUGHPUT,
        sla=SLATarget(itl_ms=30),
        strict_sla=True,
    )
    assert goal.strict_sla

    goodput = OptimizationGoal(
        target=OptimizationTarget.GOODPUT,
        sla=SLATarget(itl_ms=30),
        strict_sla=True,
    )
    assert goodput.strict_sla


def test_scalar_target_directions():
    assert OptimizationTarget.THROUGHPUT.maximize
    assert OptimizationTarget.THROUGHPUT_PER_GPU.maximize
    assert OptimizationTarget.THROUGHPUT_PER_USER.maximize
    assert OptimizationTarget.GOODPUT_PER_GPU.maximize
    assert not OptimizationTarget.E2E_LATENCY.maximize
    with pytest.raises(ValueError, match="no scalar direction"):
        _ = OptimizationTarget.PARETO.maximize


def test_pareto_defaults_and_custom_objectives():
    default = OptimizationGoal(target=OptimizationTarget.PARETO)
    custom = OptimizationGoal(
        target=OptimizationTarget.PARETO,
        pareto_objectives=[
            OptimizationTarget.THROUGHPUT_PER_USER,
            OptimizationTarget.GOODPUT_PER_GPU,
        ],
        sla=SLATarget(ttft_ms=2000, itl_ms=30),
    )

    assert default.resolved_pareto_objectives == [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    assert custom.resolved_pareto_objectives[1] is OptimizationTarget.GOODPUT_PER_GPU


@pytest.mark.parametrize(
    "goal",
    [
        {
            "target": "throughput",
            "pareto_objectives": ["throughput_per_gpu", "throughput_per_user"],
        },
        {"target": "pareto", "pareto_objectives": []},
        {"target": "pareto", "pareto_objectives": ["throughput_per_gpu"]},
        {
            "target": "pareto",
            "pareto_objectives": ["throughput_per_gpu", "throughput_per_gpu"],
        },
        {
            "target": "pareto",
            "pareto_objectives": ["goodput_per_gpu", "throughput_per_user"],
        },
    ],
)
def test_invalid_pareto_goals_are_rejected(goal):
    with pytest.raises(ValidationError):
        OptimizationGoal.model_validate(goal)


def test_kv_load_range_is_pareto_only_and_defaults_for_synthetic_pareto():
    with pytest.raises(ValidationError, match="ranged workload.kv_load_ratio"):
        SmartSearchConfig(
            search_space=_search_space(),
            workload={
                "isl": 1024,
                "osl": 1024,
                "kv_load_ratio": [0.0, 1.0],
                "num_request_ratio": 10,
            },
            goal={"target": "throughput_per_gpu"},
        )

    explicit = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.25, 0.75],
            "num_request_ratio": 10,
        },
        goal={"target": "pareto"},
    )
    defaulted = SmartSearchConfig(
        search_space=_search_space(),
        workload={"isl": 1024, "osl": 1024, "num_request_ratio": 10},
        goal={"target": "pareto"},
    )

    assert explicit.workload.kv_load_ratio_range == (0.25, 0.75)
    assert defaulted.workload.kv_load_ratio == [0.0, 1.0]


def test_scalar_kv_load_works_for_scalar_goal():
    config = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": 0.75,
            "num_request_ratio": 10,
        },
        goal={"target": "throughput_per_gpu"},
    )

    assert config.workload.kv_load_ratio == 0.75
    assert config.workload.kv_load_ratio_range is None


@pytest.mark.parametrize(
    "value",
    [[], [0.0], [0.0, 0.5, 1.0], [-0.1, 1.0], [1.0, 1.0], float("inf")],
)
def test_invalid_kv_load_ratio_is_rejected(value):
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(),
            workload={
                "isl": 1024,
                "osl": 1024,
                "kv_load_ratio": value,
                "num_request_ratio": 10,
            },
            goal={"target": "pareto"},
        )


@pytest.mark.parametrize("kwargs", [{"ttft_ms": 0}, {"itl_ms": -1}, {"e2e_ms": -5}])
def test_non_positive_sla_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        SLATarget(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rounds": 0},
        {"parallel_evals": 0},
        {"candidates_per_round": 0},
        {"max_eval_seconds": 0},
    ],
)
def test_non_positive_sweep_control_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        SweepConfig(**kwargs)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 16), (32, 16)],
)
def test_invalid_min_gpu_budget_is_rejected(minimum, maximum):
    with pytest.raises(ValidationError, match="min_gpu_budget"):
        SearchSpace(
            model_name="m",
            hardware_sku="h200_sxm",
            gpu_budget=maximum,
            min_gpu_budget=minimum,
        )


def test_valid_min_gpu_budget_is_accepted():
    space = SearchSpace(
        model_name="m",
        hardware_sku="h200_sxm",
        gpu_budget=32,
        min_gpu_budget=8,
    )

    assert space.min_gpu_budget == 8


def test_forward_model_defaults_to_op_level_for_every_role():
    space = SearchSpace(**_search_space())

    assert space.agg_forward_model == "op_level"
    assert space.prefill_forward_model == "op_level"
    assert space.decode_forward_model == "op_level"


@pytest.mark.parametrize("field", ["agg_forward_model", "prefill_forward_model", "decode_forward_model"])
def test_unknown_forward_model_is_rejected(field):
    with pytest.raises(ValidationError, match=f"{field} has invalid choice 'layerwise'"):
        SearchSpace(**_search_space(**{field: "layerwise"}))
