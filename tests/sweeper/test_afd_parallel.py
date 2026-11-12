# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD parallel topology and complete-enumeration contracts."""

import pytest

from aisimulate.sweeper import AFDInfeasible, AFDReasonCategory, AFDSearchConfig, AFDTopology, enumerate_afd_topologies


def _topology(**overrides):
    values = {
        "n_a_nodes": 1,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "a_tp_size": 1,
        "is_moe": True,
        "num_experts": 384,
        "batch_size_per_a_replica": 64,
        "num_microbatches": 3,
    }
    values.update(overrides)
    return AFDTopology.from_mapping(values)


def test_topology_derives_workers_batch_and_gpu_accounting():
    topology = _topology(n_a_nodes=2, n_f_nodes=3, a_tp_size=1, batch_size_per_a_replica=65)

    assert topology.a_replica_count == 16
    assert topology.f_rank_count == 24
    assert topology.f_gpu_count == 24
    assert topology.a_gpu_count == 16
    assert topology.f_gpu_count == 24
    assert topology.total_gpus == 40
    assert topology.microbatch_size == 22
    assert topology.total_batch_size == 1040
    assert topology.total_microbatch_size == 352
    assert topology.provenance()["gpu_accounting"]["afd_total_gpus"] == 40
    assert "source" not in topology.provenance()
    assert topology.provenance()["topology"]["comm_overhead_factor"] == 1.0
    assert topology.provenance()["topology"]["is_moe"] is True
    assert topology.provenance()["topology"]["num_experts"] == 384


def test_topology_rejects_invalid_ep_contracts():
    with pytest.raises(NotImplementedError) as dense_ep_error:
        _topology(f_moe_ep_size=2)
    assert "f_moe_tp_size" in str(dense_ep_error.value)

    with pytest.raises(ValueError, match="num_experts=10"):
        _topology(
            is_moe=True,
            num_experts=10,
            f_moe_ep_size=8,
        )


def test_default_dense_domain_rejects():
    with pytest.raises(NotImplementedError, match="Dense AFD"):
        enumerate_afd_topologies(AFDSearchConfig(total_gpus=32, gpus_per_node=8, is_moe=False))


def test_moe_domain_resolves_symbolic_ep_and_filters_expert_divisibility():
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=24,
            gpus_per_node=8,
            is_moe=True,
            num_experts=16,
            f_moe_ep_size_candidates=(3, 8, "f_gpu_count"),
            microbatch_candidates=(3,),
        )
    )
    assert [(t.n_a_nodes, t.n_f_nodes, t.a_tp_size, t.f_moe_ep_size) for t in result.candidates] == [
        (1, 1, 1, 8),
        (1, 2, 1, 16),
        (2, 1, 1, 8),
    ]
    assert result.provenance["complete"] is True


@pytest.mark.parametrize("budget", [16, 24, 32])
@pytest.mark.parametrize("f_width", [4, 8])
def test_fixed_ep_keeps_feasible_topologies_as_gpu_budget_grows(budget, f_width):
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=budget,
            gpus_per_node=8,
            f_gpus_per_node=f_width,
            is_moe=True,
            num_experts=128,
            f_moe_ep_size_candidates=(8,),
            batch_size_per_a_replica_candidates=(18,),
            microbatch_candidates=(3,),
        )
    )
    assert [(t.n_a_nodes, t.n_f_nodes, t.f_moe_ep_size) for t in result.candidates] == [
        (a, 8 // f_width, 8) for a in range(1, (budget - 8) // 8 + 1)
    ]
    assert result.provenance["complete"] is True


def test_fixed_ep_without_supported_f_width_reports_no_feasible_topology():
    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=24,
                gpus_per_node=8,
                is_moe=True,
                num_experts=128,
                f_moe_ep_size_candidates=(2,),
            )
        )
    assert error.value.category is AFDReasonCategory.NO_FEASIBLE_TOPOLOGY


def test_search_candidate_types_are_strict():
    with pytest.raises(AFDInfeasible, match="a_tp_size_candidates must be a positive integer"):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=("8",),
        )

    with pytest.raises(AFDInfeasible, match="f_moe_ep_size_candidates accepts positive integers"):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            f_moe_ep_size_candidates=(True,),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_experts", 8),
        ("f_moe_ep_size_candidates", (2,)),
        ("f_moe_ep_size_candidates", (1, 2)),
        ("f_moe_ep_size_candidates", ("n_f_nodes",)),
        ("f_moe_ep_size_candidates", ("f_gpu_count",)),
    ],
)
def test_dense_search_rejects_moe_only_settings(field, value):
    with pytest.raises(AFDInfeasible) as error:
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=False,
            **{field: value},
        )

    assert error.value.category is AFDReasonCategory.EXPERT_DIVISIBILITY
    assert error.value.provenance["field"] == field
    assert error.value.provenance["value"] == value


def test_dense_search_rejects_even_single_expert_parallelism():
    with pytest.raises(NotImplementedError, match="Dense AFD"):
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=False,
                f_moe_ep_size_candidates=(1,),
            )
        )


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("a_tp_size_candidates", (2, 2)),
        ("batch_size_per_a_replica_candidates", (64, 64)),
        ("f_moe_ep_size_candidates", (1, 1)),
        ("microbatch_candidates", (3, 3)),
    ],
)
def test_search_rejects_duplicate_explicit_candidates(field, values):
    with pytest.raises(AFDInfeasible, match=f"{field} must not contain duplicates"):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=field == "f_moe_ep_size_candidates",
            **{field: values},
        )


def test_search_rejects_nondivisible_explicit_tp_candidate():
    with pytest.raises(AFDInfeasible, match="a_tp_size_candidates must divide gpus_per_node") as error:
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=(2, 3),
        )

    assert error.value.provenance["invalid_candidates"] == [3]


def test_pinned_domain_is_lossless_and_honors_budget():
    pinned = _topology(n_a_nodes=1, n_f_nodes=2)
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=24,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            pinned_topologies=(pinned,),
        )
    )
    assert result.candidates == (pinned,)
    assert result.provenance["domain"] == "pinned"
    assert result.provenance["source"] == "AFDSearchConfig.pinned_topologies"
    assert "candidate_order" not in result.provenance

    with pytest.raises(AFDInfeasible) as budget_error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=True,
                num_experts=384,
                pinned_topologies=(pinned,),
            )
        )
    assert budget_error.value.category is AFDReasonCategory.GPU_BUDGET


def test_search_rejects_duplicate_pinned_topologies():
    pinned = _topology()

    with pytest.raises(AFDInfeasible, match="pinned_topologies must not contain duplicates") as error:
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            pinned_topologies=(pinned, _topology()),
        )

    assert error.value.category is AFDReasonCategory.INVALID_TOPOLOGY
    assert error.value.provenance["field"] == "pinned_topologies"
    assert error.value.provenance["values"] == [
        pinned.provenance(),
        pinned.provenance(),
    ]


def test_pinned_domain_accepts_budget_with_partial_node_remainder():
    pinned = _topology()
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=20,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            pinned_topologies=(pinned,),
        )
    )

    assert result.candidates == (pinned,)
    assert result.generated_count == 1
    assert result.provenance["complete"] is True


@pytest.mark.parametrize(
    ("topology_overrides", "config_overrides", "field"),
    [
        ({"comm_overhead_factor": 2.0}, {}, "comm_overhead_factor"),
        ({"pipeline_layout": "two_stage"}, {}, "pipeline_layout"),
    ],
)
def test_pinned_domain_rejects_search_contract_mismatches(topology_overrides, config_overrides, field):
    pinned = _topology(**topology_overrides)

    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=True,
                num_experts=384,
                pinned_topologies=(pinned,),
                **config_overrides,
            )
        )

    assert error.value.category is AFDReasonCategory.INVALID_TOPOLOGY
    assert field in error.value.provenance["mismatches"]


def test_candidate_limit_requires_a_complete_domain():
    error_config = AFDSearchConfig(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=True,
        num_experts=384,
        max_candidates=2,
    )
    with pytest.raises(AFDInfeasible) as overflow:
        enumerate_afd_topologies(error_config)
    assert overflow.value.category is AFDReasonCategory.CANDIDATE_LIMIT
    assert overflow.value.provenance["generated_count"] > 2
    assert overflow.value.provenance["count_is_lower_bound"] is True


def test_search_requires_two_node_minimum_with_actionable_budget_reason():
    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(AFDSearchConfig(total_gpus=8, gpus_per_node=8, is_moe=True, num_experts=384))
    assert error.value.category is AFDReasonCategory.GPU_BUDGET
    assert "at least 16 GPUs" in error.value.detail


def test_generated_domain_applies_minimum_budget_and_inclusive_af_ratio():
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=32,
            min_gpu_budget=24,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=(1,),
            batch_size_per_a_replica_candidates=(1,),
            microbatch_candidates=(1,),
            max_af_ratio=1.0,
        )
    )

    node_pairs = {(item.n_a_nodes, item.n_f_nodes) for item in result.candidates}
    assert (1, 1) not in node_pairs
    assert (2, 2) in node_pairs
    assert all(item.total_gpus >= 24 for item in result.candidates)
    assert all(item.af_node_ratio <= 1.0 for item in result.candidates)
    assert result.rejection_counts[AFDReasonCategory.GPU_BUDGET.value] == 1
    assert result.rejection_counts["af_ratio"] == 2


@pytest.mark.parametrize(
    ("total_gpus", "expected_gpu_counts", "remainder_gpus"),
    [(16, {16}, 0), (20, {16}, 4), (23, {16}, 7), (24, {16, 24}, 0)],
)
def test_generated_domain_treats_total_gpus_as_an_upper_bound(total_gpus, expected_gpu_counts, remainder_gpus):
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=total_gpus,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=(1,),
            batch_size_per_a_replica_candidates=(1,),
            microbatch_candidates=(1,),
        )
    )

    assert {item.total_gpus for item in result.candidates} == expected_gpu_counts
    assert result.provenance["complete"] is True
    assert result.provenance["gpu_budget"] == {
        "granularity": "node",
        "total_gpus": total_gpus,
        "gpus_per_node": 8,
        "f_gpus_per_node": 8,
        "usable_gpus": max(expected_gpu_counts),
        "remainder_gpus": remainder_gpus,
    }


def test_search_has_one_candidate_per_microbatch_count():
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=(1,),
            batch_size_per_a_replica_candidates=(8,),
            microbatch_candidates=(1, 2, 3, 4),
        )
    )
    assert [item.num_microbatches for item in result.candidates] == [1, 2, 3, 4]
    assert result.generated_count == 4
    assert all("pipeline_model" not in item.provenance()["topology"] for item in result.candidates)
    assert "pipeline_model" not in result.provenance["candidate_order"]


@pytest.mark.parametrize("is_moe", [False, True])
def test_f_routing_is_not_filtered_or_fallen_back_during_enumeration(is_moe):
    cfg = AFDSearchConfig(total_gpus=16, gpus_per_node=8, is_moe=is_moe, routing_location="f")
    with pytest.raises(NotImplementedError, match="routing_location='f' is not implemented"):
        enumerate_afd_topologies(cfg)
    cfg = AFDSearchConfig(
        total_gpus=16,
        gpus_per_node=8,
        is_moe=True,
        num_experts=384,
        pinned_topologies=(_topology(routing_location="f"),),
    )
    with pytest.raises(NotImplementedError, match="routing_location='f'"):
        enumerate_afd_topologies(cfg)


@pytest.mark.parametrize("a_width,f_width", [(8, 4), (4, 8), (8, 8)])
def test_heterogeneous_width_domain_matches_independent_node_budget(a_width, f_width):
    budget = 28
    domain = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=budget,
            gpus_per_node=a_width,
            f_gpus_per_node=f_width,
            is_moe=True,
            num_experts=384,
            a_tp_size_candidates=(1, 2),
            batch_size_per_a_replica_candidates=(17,),
            microbatch_candidates=(3,),
        )
    )
    expected = [
        (a, f, tp)
        for a in range(1, budget // a_width + 1)
        for f in range(1, budget // f_width + 1)
        if a * a_width + f * f_width <= budget and a / f <= 4 and 384 % (f * f_width) == 0
        for tp in (1, 2)
    ]
    assert [(t.n_a_nodes, t.n_f_nodes, t.a_tp_size) for t in domain.candidates] == expected
    for t in domain.candidates:
        assert t.total_gpus == t.n_a_nodes * a_width + t.n_f_nodes * f_width
        assert t.f_moe_ep_size == t.n_f_nodes * f_width
        assert t.a_replica_count == t.n_a_nodes * a_width // t.a_tp_size
        assert AFDTopology.from_mapping(t.provenance()["topology"]) == t


def test_heterogeneous_minimum_and_pinned_width_validation():
    config = dict(gpus_per_node=8, f_gpus_per_node=4, is_moe=True, num_experts=384)
    assert len(enumerate_afd_topologies(AFDSearchConfig(total_gpus=12, **config)).candidates) == 3
    with pytest.raises(AFDInfeasible, match="at least 12 GPUs"):
        enumerate_afd_topologies(AFDSearchConfig(total_gpus=11, **config))
    pinned = _topology(f_gpus_per_node=8)
    with pytest.raises(AFDInfeasible, match="search hardware"):
        enumerate_afd_topologies(AFDSearchConfig(total_gpus=32, pinned_topologies=(pinned,), **config))


@pytest.mark.parametrize("width", [0, -1, True, 4.5])
def test_f_node_width_must_be_a_positive_integer(width):
    with pytest.raises(AFDInfeasible, match="f_gpus_per_node"):
        _topology(f_gpus_per_node=width)


def test_old_topology_mapping_inherits_node_width_and_checks_saved_counts():
    old = _topology().provenance()["topology"]
    old.pop("f_gpus_per_node")
    assert AFDTopology.from_mapping(old).f_gpus_per_node == 8
    old["f_gpu_count"] = 4
    with pytest.raises(ValueError, match="f_gpu_count"):
        AFDTopology.from_mapping(old)


def test_microbatch_filter_precedes_candidate_limit_and_keeps_equality():
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            num_experts=128,
            batch_size_per_a_replica_candidates=(1, 3),
            microbatch_candidates=(1, 3, 4),
            max_candidates=3,
        )
    )
    assert [(t.batch_size_per_a_replica, t.num_microbatches) for t in result.candidates] == [(1, 1), (3, 1), (3, 3)]
    assert result.generated_count == 3
    assert result.rejection_counts["microbatch_geometry"] == 3


def test_all_invalid_microbatches_report_geometry_rejections():
    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=True,
                num_experts=128,
                batch_size_per_a_replica_candidates=(1,),
                microbatch_candidates=(2, 3),
            )
        )
    assert error.value.category == AFDReasonCategory.NO_FEASIBLE_TOPOLOGY
    assert error.value.provenance["rejection_counts"]["microbatch_geometry"] == 2


def test_pinned_microbatch_geometry_rejects_at_construction():
    with pytest.raises(AFDInfeasible, match="num_microbatches cannot exceed"):
        _topology(batch_size_per_a_replica=1, num_microbatches=3)
