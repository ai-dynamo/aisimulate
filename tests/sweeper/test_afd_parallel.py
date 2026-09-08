# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD parallel topology and complete-enumeration contracts."""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aiconfigurator.sdk.task_v2 import build_afd_parallel_lists
from aisimulate.config import CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper import (
    AFDInfeasible,
    AFDReasonCategory,
    AFDSearchConfig,
    AFDTopology,
    enumerate_afd_topologies,
)

_AFD_MIGRATION_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "cli" / "migrate-from-aiconfigurator.md"


def _documented_afd_recommendation() -> dict:
    text = _AFD_MIGRATION_GUIDE.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- afd-migration-contract-start -->\s*```yaml\s*(.*?)\s*```\s*"
        r"<!-- afd-migration-contract-end -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "AFD migration guide must contain one marked YAML contract"
    payload = yaml.safe_load(match.group(1))
    assert isinstance(payload, dict)
    return payload


def _topology(**overrides):
    values = {
        "n_a_nodes": 1,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "tp_a": 2,
        "a_batch_size": 64,
        "num_microbatches": 3,
        "pipeline_model": "optimistic",
        "phase": "decode",
        "combined_with_pd": False,
    }
    values.update(overrides)
    return AFDTopology(**values)


def test_documented_afd_migration_contract_is_well_formed():
    payload = _documented_afd_recommendation()

    assert payload["traffic"]["source"] == {
        "type": "synthetic",
        "input_tokens": 1024,
        "output_tokens": 128,
    }
    assert payload["engine"] == {
        "mode": "afd",
        "model": "Qwen/Qwen3-32B",
        "hardware": "h200_sxm",
        "backend": "trtllm",
        "afd": {
            "phase": "decode",
            "combined_with_pd": True,
        },
    }
    assert payload["evaluation"]["sla"] == {
        "ttft_ms": 800,
        "itl_ms": 30,
    }
    assert payload["optimization"] == {
        "target": "throughput_per_gpu",
        "strict_sla": True,
        "constraints": {"max_candidate_gpus": 32},
    }


@pytest.mark.xfail(
    reason=(
        "AIC-1775 PR 17 defines the AFD Sweeper-core contract; the public "
        "recommendation schema and lowering land in a later PR"
    ),
    raises=ValidationError,
    strict=True,
)
def test_documented_afd_migration_contract_lowers_to_sweeper():
    config = CoreRecommendationConfig.model_validate(_documented_afd_recommendation())

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.deployment_mode == ["afd"]


def test_topology_derives_workers_batch_and_gpu_accounting():
    topology = _topology(n_a_nodes=2, n_f_nodes=3, tp_a=4, a_batch_size=65)

    assert topology.attention_workers == 4
    assert topology.ffn_workers == 24
    assert topology.ffn_tp == 24
    assert topology.attention_gpus == 16
    assert topology.ffn_gpus == 24
    assert topology.total_gpus == 40
    assert topology.microbatch_size == 22
    assert topology.total_batch_size == 260
    assert topology.total_microbatch_size == 88
    assert topology.provenance()["gpu_accounting"]["afd_total_gpus"] == 40
    assert topology.provenance()["topology"]["comm_overhead_factor"] == 1.0
    assert topology.provenance()["topology"]["is_moe"] is False
    assert topology.provenance()["topology"]["num_experts"] == 0


def test_topology_rejects_invalid_phase_and_ep_contracts():
    with pytest.raises(AFDInfeasible) as phase_error:
        _topology(phase="both", combined_with_pd=True)
    assert phase_error.value.category is AFDReasonCategory.INCOMPATIBLE_PHASE

    with pytest.raises(AFDInfeasible) as dense_ep_error:
        _topology(f_moe_ep_size=2)
    assert dense_ep_error.value.category is AFDReasonCategory.EXPERT_DIVISIBILITY

    with pytest.raises(AFDInfeasible, match="num_experts=10"):
        _topology(
            is_moe=True,
            num_experts=10,
            f_moe_ep_size=8,
        )


def test_default_dense_domain_matches_legacy_candidate_order():
    config = AFDSearchConfig(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=False,
        a_batch_size_candidates=(128,),
    )

    result = enumerate_afd_topologies(config)
    actual = [
        (
            item.n_a_nodes,
            item.n_f_nodes,
            item.tp_a,
            item.f_moe_ep_size,
            item.num_microbatches,
            item.pipeline_model.value,
        )
        for item in result.candidates
    ]
    expected = build_afd_parallel_lists(
        total_gpus=32,
        gpus_per_node=8,
        is_moe=False,
    )

    assert actual == expected
    assert result.provenance["complete"] is True
    assert result.provenance["source"].endswith("build_afd_parallel_lists")


def test_moe_domain_resolves_symbolic_ep_and_filters_expert_divisibility():
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=24,
            gpus_per_node=8,
            is_moe=True,
            num_experts=16,
            tp_a_candidates=(8,),
            f_moe_ep_size_candidates=(3, "n_f_nodes", "ffn_tp"),
            microbatch_candidates=(3,),
            pipeline_model_candidates=("optimistic",),
        )
    )

    assert result.candidates
    assert all(item.ffn_tp % item.f_moe_ep_size == 0 for item in result.candidates)
    assert all(16 % item.f_moe_ep_size == 0 for item in result.candidates)
    assert {item.f_moe_ep_size for item in result.candidates} <= {1, 2, 8, 16}
    actual = [
        (
            item.n_a_nodes,
            item.n_f_nodes,
            item.tp_a,
            item.f_moe_ep_size,
            item.num_microbatches,
            item.pipeline_model.value,
        )
        for item in result.candidates
    ]
    expected = build_afd_parallel_lists(
        total_gpus=24,
        gpus_per_node=8,
        is_moe=True,
        num_experts=16,
        search_config={
            "tp_a_list": [8],
            "f_moe_ep_size_list": [3, "n_f_nodes", "tp_f"],
            "microbatch_list": [3],
            "pipeline_model_list": ["optimistic"],
        },
    )
    assert actual == expected


def test_search_candidate_types_are_strict():
    with pytest.raises(AFDInfeasible, match="tp_a_candidates must be a positive integer"):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=False,
            tp_a_candidates=("8",),
        )

    with pytest.raises(AFDInfeasible, match="f_moe_ep_size_candidates accepts positive integers"):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=True,
            f_moe_ep_size_candidates=(True,),
        )


def test_pinned_domain_is_lossless_and_honors_budget():
    pinned = _topology(n_a_nodes=1, n_f_nodes=2)
    result = enumerate_afd_topologies(
        AFDSearchConfig(
            total_gpus=24,
            gpus_per_node=8,
            is_moe=False,
            pinned_topologies=(pinned,),
            combined_with_pd=False,
        )
    )
    assert result.candidates == (pinned,)
    assert result.provenance["domain"] == "pinned"

    with pytest.raises(AFDInfeasible) as budget_error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=False,
                pinned_topologies=(pinned,),
                combined_with_pd=False,
            )
        )
    assert budget_error.value.category is AFDReasonCategory.GPU_BUDGET


@pytest.mark.parametrize(
    ("topology_overrides", "config_overrides", "field"),
    [
        ({"phase": "prefill"}, {}, "phase"),
        ({"combined_with_pd": False}, {}, "combined_with_pd"),
        ({"comm_overhead_factor": 2.0}, {}, "comm_overhead_factor"),
        ({"boundary_on_attn": False}, {}, "boundary_on_attn"),
    ],
)
def test_pinned_domain_rejects_search_contract_mismatches(topology_overrides, config_overrides, field):
    pinned = _topology(**topology_overrides)

    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(
            AFDSearchConfig(
                total_gpus=16,
                gpus_per_node=8,
                is_moe=False,
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
        is_moe=False,
        max_candidates=2,
    )
    with pytest.raises(AFDInfeasible) as overflow:
        enumerate_afd_topologies(error_config)
    assert overflow.value.category is AFDReasonCategory.CANDIDATE_LIMIT
    assert overflow.value.provenance["generated_count"] > 2
    assert overflow.value.provenance["count_is_lower_bound"] is True


def test_search_requires_two_node_minimum_with_actionable_budget_reason():
    with pytest.raises(AFDInfeasible) as error:
        enumerate_afd_topologies(AFDSearchConfig(total_gpus=8, gpus_per_node=8, is_moe=False))
    assert error.value.category is AFDReasonCategory.GPU_BUDGET
    assert "at least 16 GPUs" in error.value.detail
