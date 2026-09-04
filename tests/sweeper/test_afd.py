# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class AFD topology, evaluator, and rate-matching contracts."""

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aiconfigurator.sdk.task_v2 import build_afd_parallel_lists
from aisimulate.config import CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.sweeper import (
    AFDCompanionOption,
    AFDInfeasible,
    AFDLayerTimes,
    AFDReasonCategory,
    AFDSearchConfig,
    AFDTopology,
    RunnerCapabilities,
    enumerate_afd_topologies,
    evaluate_afd_phase,
    evaluate_pure_afd,
    rate_match_afd_with_pd,
    require_afd_adapter_support,
    require_afd_runner_support,
)


_AFD_MIGRATION_GUIDE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "cli"
    / "migrate-from-aiconfigurator.md"
)


def _documented_afd_recommendation() -> dict:
    text = _AFD_MIGRATION_GUIDE.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- afd-migration-contract-start -->\s*```yaml\s*(.*?)\s*```\s*"
        r"<!-- afd-migration-contract-end -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, (
        "AFD migration guide must contain one marked YAML contract"
    )
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


def _times(phase="decode", **overrides):
    values = {
        "phase": phase,
        "attention_ms": 1.0,
        "ffn_ms": 2.0,
        "a_to_f_ms": 0.25,
        "f_to_a_ms": 0.25,
        "num_layers": 2,
        "provenance": {"database": "fixture-v1"},
    }
    values.update(overrides)
    return AFDLayerTimes(**values)


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
    with pytest.raises(
        AFDInfeasible, match="tp_a_candidates must be a positive integer"
    ):
        AFDSearchConfig(
            total_gpus=16,
            gpus_per_node=8,
            is_moe=False,
            tp_a_candidates=("8",),
        )

    with pytest.raises(
        AFDInfeasible, match="f_moe_ep_size_candidates accepts positive integers"
    ):
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
def test_pinned_domain_rejects_search_contract_mismatches(
    topology_overrides, config_overrides, field
):
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
        enumerate_afd_topologies(
            AFDSearchConfig(total_gpus=8, gpus_per_node=8, is_moe=False)
        )
    assert error.value.category is AFDReasonCategory.GPU_BUDGET
    assert "at least 16 GPUs" in error.value.detail


def test_pipeline_evaluator_matches_legacy_optimistic_formula():
    topology = _topology()
    result = evaluate_afd_phase(
        topology,
        _times(),
        input_length=128,
        output_length=16,
    )

    # fill=3.5; cycle=max(1,2,.5)=2; global step=3.5+2*(3*2-1)
    assert result.pipeline_fill_ms == pytest.approx(3.5)
    assert result.cycle_ms == pytest.approx(2.0)
    assert result.step_latency_ms == pytest.approx(13.5)
    assert result.communication_hidden is True
    assert result.effective_pipeline_model.value == "optimistic"
    assert result.balance_ratio == pytest.approx(0.5)
    assert result.tokens_per_second == pytest.approx(256 / 0.0135)
    assert result.sequence_rate == pytest.approx((256 / 0.0135) / 16)


def test_optimistic_pipeline_falls_back_when_microbatch_count_is_too_small():
    topology = _topology(num_microbatches=2)
    result = evaluate_afd_phase(
        topology,
        _times(attention_ms=1, ffn_ms=1, a_to_f_ms=1, f_to_a_ms=1),
        input_length=128,
        output_length=16,
    )

    assert result.requested_pipeline_model.value == "optimistic"
    assert result.effective_pipeline_model.value == "conservative"
    assert result.communication_hidden is False
    assert result.cycle_ms == pytest.approx(2.0)
    assert result.step_latency_ms == pytest.approx(10.0)


def test_communication_overhead_is_applied_and_provenanced():
    result = evaluate_afd_phase(
        _topology(comm_overhead_factor=2.0),
        _times(),
        input_length=128,
        output_length=16,
    )

    assert result.provenance["layer_times"]["a_to_f_ms"] == pytest.approx(0.5)
    assert result.provenance["layer_times"]["f_to_a_ms"] == pytest.approx(0.5)


def test_pure_both_phase_rate_matches_without_double_counting_gpus():
    topology = _topology(phase="both", combined_with_pd=False)
    result = evaluate_pure_afd(
        topology,
        [_times("prefill"), _times("decode")],
        input_length=128,
        output_length=16,
        prefill_degradation=0.9,
        decode_degradation=0.95,
        ttft_correction_factor=1.8,
        decode_latency_correction=1.2,
    )

    assert result.total_gpus == topology.total_gpus == 16
    assert result.companion_gpus == 0
    assert set(result.phase_evaluations) == {"prefill", "decode"}
    assert result.ttft_ms == pytest.approx(13.5 * 1.8)
    assert result.tpot_ms == pytest.approx(13.5 * 1.2)
    assert result.sequence_rate == pytest.approx(
        min((256 / 0.0135) * 0.9, ((256 / (0.0135 * 1.2)) / 16) * 0.95)
    )
    json.dumps(result.as_dict(), sort_keys=True)


def test_pure_prefill_does_not_report_generation_throughput():
    topology = _topology(phase="prefill", combined_with_pd=False)
    result = evaluate_pure_afd(
        topology,
        [_times("prefill")],
        input_length=128,
        output_length=16,
    )

    assert result.sequence_rate > 0
    assert result.phase_evaluations["prefill"].tokens_per_second > 0
    assert result.tokens_per_second == 0
    assert result.tokens_per_second_per_gpu == 0


def test_pure_afd_rejects_duplicate_and_unexpected_phase_measurements():
    decode = _times("decode")
    topology = _topology(phase="decode", combined_with_pd=False)

    with pytest.raises(AFDInfeasible, match="duplicate") as duplicate:
        evaluate_pure_afd(
            topology,
            [decode, decode],
            input_length=128,
            output_length=16,
        )
    assert duplicate.value.category is AFDReasonCategory.INVALID_MEASUREMENT

    with pytest.raises(AFDInfeasible, match="unexpected") as unexpected:
        evaluate_pure_afd(
            topology,
            [decode, _times("prefill")],
            input_length=128,
            output_length=16,
        )
    assert unexpected.value.category is AFDReasonCategory.INVALID_MEASUREMENT


def test_decode_afd_rate_matches_static_prefill_and_accounts_all_gpus():
    topology = _topology(combined_with_pd=True)
    options = [
        AFDCompanionOption(
            phase="prefill",
            sequence_rate_per_worker=1000,
            latency_ms=10,
            gpus_per_worker=4,
            parallel_config={"tp": 4, "batch_size": 8},
            provenance={"system": "h200_sxm"},
        )
    ]

    result = rate_match_afd_with_pd(
        topology,
        _times(),
        options,
        input_length=128,
        output_length=16,
        total_gpu_budget=24,
    )

    assert result.companion_workers in {1, 2}
    assert result.companion_gpus == result.companion_workers * 4
    assert result.total_gpus == topology.total_gpus + result.companion_gpus
    assert result.ttft_ms == pytest.approx(18.0)
    assert result.tpot_ms == pytest.approx(13.5)
    assert result.sequence_rate <= result.companion_workers * 1000 * 0.9
    assert result.provenance["gpu_accounting"] == {
        "attention_gpus": 8,
        "ffn_gpus": 8,
        "companion_gpus": result.companion_gpus,
        "total_gpus": result.total_gpus,
    }


def test_prefill_afd_can_rate_match_static_decode_companion():
    topology = _topology(phase="prefill", combined_with_pd=True)
    option = AFDCompanionOption(
        phase="decode",
        sequence_rate_per_worker=500,
        latency_ms=4,
        gpus_per_worker=2,
    )
    result = rate_match_afd_with_pd(
        topology,
        _times("prefill"),
        [option],
        input_length=128,
        output_length=16,
        total_gpu_budget=32,
        ttft_correction_factor=1.5,
        decode_latency_correction=1.25,
    )

    assert result.ttft_ms == pytest.approx(13.5 * 1.5)
    assert result.tpot_ms == pytest.approx(5.0)
    assert result.companion.phase.value == "decode"


def test_rate_match_reports_rejection_counts_when_no_companion_is_feasible():
    topology = _topology(combined_with_pd=True)
    option = AFDCompanionOption(
        phase="prefill",
        sequence_rate_per_worker=100,
        latency_ms=10,
        gpus_per_worker=8,
    )

    with pytest.raises(AFDInfeasible) as error:
        rate_match_afd_with_pd(
            topology,
            _times(),
            [option],
            input_length=128,
            output_length=16,
            total_gpu_budget=24,
            target_ttft_ms=5,
        )
    assert error.value.category is AFDReasonCategory.NO_FEASIBLE_COMPANION
    assert error.value.provenance["rejection_counts"]["latency_sla"] > 0


def test_companion_candidate_limit_requires_a_complete_domain():
    topology = _topology(combined_with_pd=True)
    option = AFDCompanionOption(
        phase="prefill",
        sequence_rate_per_worker=1000,
        latency_ms=10,
        gpus_per_worker=4,
    )
    options = [option, option]

    with pytest.raises(AFDInfeasible) as overflow:
        rate_match_afd_with_pd(
            topology,
            _times(),
            options,
            input_length=128,
            output_length=16,
            max_companion_candidates=1,
        )
    assert overflow.value.category is AFDReasonCategory.CANDIDATE_LIMIT
    assert overflow.value.provenance["domain"] == "companion"


def test_companion_worker_expansion_is_bounded_and_never_overflows():
    topology = _topology(combined_with_pd=True)
    option = AFDCompanionOption(
        phase="prefill",
        sequence_rate_per_worker=5e-324,
        latency_ms=10,
        gpus_per_worker=1,
    )

    with pytest.raises(AFDInfeasible) as overflow:
        rate_match_afd_with_pd(
            topology,
            _times(),
            [option],
            input_length=128,
            output_length=16,
        )
    assert overflow.value.category is AFDReasonCategory.CANDIDATE_LIMIT

    bounded = rate_match_afd_with_pd(
        topology,
        _times(),
        [option],
        input_length=128,
        output_length=16,
        max_companion_workers=2,
    )
    assert bounded.companion_workers == 2
    assert bounded.provenance["companion_domain"]["evaluated_candidates"] == 2


def test_unsupported_adapters_and_runners_fail_closed():
    topology = _topology(combined_with_pd=True)

    with pytest.raises(AFDInfeasible) as adapter_error:
        require_afd_adapter_support("generator", {"agg", "disagg"}, topology)
    assert adapter_error.value.category is AFDReasonCategory.UNSUPPORTED_ADAPTER
    assert adapter_error.value.provenance["required_topology"] == "afd+pd"

    with pytest.raises(AFDInfeasible) as runner_error:
        require_afd_runner_support(
            RunnerCapabilities(supported_backend_topologies=(("vllm", "afd"),)),
            "vllm",
            topology,
        )
    assert runner_error.value.category is AFDReasonCategory.UNSUPPORTED_ADAPTER

    require_afd_adapter_support("generator", {"afd+pd"}, topology)
    require_afd_runner_support(
        RunnerCapabilities(supported_backend_topologies=(("vllm", "afd+pd"),)),
        "vllm",
        topology,
    )
