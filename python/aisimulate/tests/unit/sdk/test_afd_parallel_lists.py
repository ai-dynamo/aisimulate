# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AFD default-mode search space enumeration."""

import pytest

from aiconfigurator.sdk.task_v2 import build_afd_parallel_lists

pytestmark = pytest.mark.unit


def test_dense_candidates_respect_budget_and_divisibility():
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    assert candidates
    for n_a, n_f, tp_a, f_ep, mb, pipe in candidates:
        assert n_a >= 1 and n_f >= 1
        assert (n_a + n_f) * 8 <= 32
        assert 8 % tp_a == 0
        assert f_ep == 1  # dense models never shard experts
        assert mb in (2, 3, 4)
        assert pipe in ("optimistic", "conservative")


def test_moe_expert_divisibility():
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=True, num_experts=256)
    assert candidates
    for _n_a, n_f, _tp_a, f_ep, _mb, _pipe in candidates:
        tp_f = n_f * 8
        assert tp_f % f_ep == 0
        assert 256 % f_ep == 0


def test_partial_node_splits_are_enumerated():
    """Combined-with-PD needs headroom: splits using < all nodes must exist."""
    candidates = build_afd_parallel_lists(total_gpus=32, gpus_per_node=8, is_moe=False)
    used_nodes = {n_a + n_f for n_a, n_f, *_ in candidates}
    assert {2, 3, 4} <= used_nodes


def test_skewed_splits_are_pruned():
    candidates = build_afd_parallel_lists(total_gpus=64, gpus_per_node=8, is_moe=False)
    assert all(n_a / n_f <= 4 for n_a, n_f, *_ in candidates)


def test_search_config_controls_candidate_axes():
    candidates = build_afd_parallel_lists(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        search_config={
            "tp_a_list": [4],
            "microbatch_list": [3],
            "pipeline_model_list": ["optimistic"],
            "f_moe_ep_size_list": [1, "n_f_nodes"],
            "max_af_ratio": 3,
        },
    )

    assert candidates
    for n_a, n_f, tp_a, f_ep, mb, pipe in candidates:
        assert n_a / n_f <= 3
        assert tp_a == 4
        assert f_ep in {1, n_f}
        assert mb == 3
        assert pipe == "optimistic"


def test_search_config_errors_when_candidate_count_exceeds_limit():
    with pytest.raises(ValueError, match="max_candidates=1"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"max_candidates": 1},
        )


def test_search_config_can_truncate_candidate_overflow():
    candidates = build_afd_parallel_lists(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=False,
        search_config={"max_candidates": 1, "candidate_overflow": "truncate"},
    )

    assert len(candidates) == 1


def test_search_config_rejects_invalid_candidate_limit():
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"max_candidates": 0},
        )


def test_search_config_rejects_invalid_overflow_policy():
    with pytest.raises(ValueError, match="candidate_overflow must be 'error' or 'truncate'"):
        build_afd_parallel_lists(
            total_gpus=32,
            gpus_per_node=8,
            is_moe=False,
            search_config={"candidate_overflow": "ignore"},
        )


def test_default_limit_covers_128_gpu_dense_search():
    candidates = build_afd_parallel_lists(total_gpus=128, gpus_per_node=8, is_moe=False)

    assert len(candidates) == 2040


def test_default_limit_covers_96_gpu_moe_search():
    candidates = build_afd_parallel_lists(
        total_gpus=96,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
    )

    assert len(candidates) > 2000


def test_single_node_returns_empty():
    assert build_afd_parallel_lists(total_gpus=8, gpus_per_node=8, is_moe=True, num_experts=64) == []


def test_invalid_inputs_return_empty():
    assert build_afd_parallel_lists(total_gpus=0, gpus_per_node=8, is_moe=False) == []
    assert build_afd_parallel_lists(total_gpus=16, gpus_per_node=0, is_moe=False) == []
    assert build_afd_parallel_lists(total_gpus=16, gpus_per_node=8, is_moe=False, f_gpus_per_node=0) == []


def test_hetero_wide_f_pool_budgets_real_footprint():
    """F nodes wider than A nodes (gb200 A + b200 F): budget on GPUs, not node count."""
    candidates = build_afd_parallel_lists(total_gpus=72, gpus_per_node=4, is_moe=False, f_gpus_per_node=8)
    assert candidates
    for n_a, n_f, *_ in candidates:
        assert n_a * 4 + n_f * 8 <= 72
    # 10 A nodes + 8 F nodes fits the old node-count budget (10+8 <= 72//4)
    # but its real footprint is 40 + 64 = 104 GPUs.
    assert not any(n_a == 10 and n_f == 8 for n_a, n_f, *_ in candidates)
    # The feasible split the node-count budget also admitted.
    assert any(n_a == 10 and n_f == 4 for n_a, n_f, *_ in candidates)  # 40 + 32 = 72


def test_hetero_narrow_f_pool_is_not_under_enumerated():
    """F nodes narrower than A nodes (h200 A + gb200 F): splits beyond the A-node count exist."""
    candidates = build_afd_parallel_lists(total_gpus=72, gpus_per_node=8, is_moe=False, f_gpus_per_node=4)
    assert candidates
    for n_a, n_f, *_ in candidates:
        assert n_a * 8 + n_f * 4 <= 72
    # Feasible on the real footprint but unreachable under the old
    # node-count budget (n_a + n_f <= 72//8 = 9).
    assert any(n_a == 1 and n_f == 16 for n_a, n_f, *_ in candidates)  # 8 + 64 = 72
    assert any(n_a == 5 and n_f == 8 for n_a, n_f, *_ in candidates)  # 40 + 32 = 72


def test_hetero_moe_ep_uses_f_pool_width():
    """The 'tp_f' EP candidate and divisibility checks use the F pool's own width."""
    candidates = build_afd_parallel_lists(
        total_gpus=72,
        gpus_per_node=4,
        is_moe=True,
        num_experts=256,
        f_gpus_per_node=8,
    )
    assert candidates
    # tp_f = n_f * 8, so n_f=2 offers the full-width EP=16 candidate (the old
    # single-width derivation capped it at n_f * 4 = 8).
    assert any(n_f == 2 and f_ep == 16 for _n_a, n_f, _tp_a, f_ep, *_ in candidates)
    for _n_a, n_f, _tp_a, f_ep, _mb, _pipe in candidates:
        assert (n_f * 8) % f_ep == 0


def test_hetero_defaults_to_homogeneous_when_unset():
    homogeneous = build_afd_parallel_lists(total_gpus=64, gpus_per_node=8, is_moe=True, num_experts=256)
    explicit = build_afd_parallel_lists(
        total_gpus=64,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        f_gpus_per_node=8,
    )
    assert homogeneous == explicit
