# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend branch enumeration and runner-capability filtering."""

from pathlib import Path

import pytest

from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.kv_estimate import NoPerfDatabase
from aisimulate.sweeper.model_hw import NoViableParallelConfig
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.replay import RunnerCapabilities
from aisimulate.sweeper.sampler import ExhaustiveBranchSampler
from aisimulate.sweeper.search_space import branch_knob_choices, enumerate_branches

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")

_AGG_CFG = ReplicaParallelConfig(
    ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
)
_AGG_ALT_CFG = ReplicaParallelConfig(
    ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1
)
_DISAGG_DP1_CFG = DisaggParallelConfig(prefill=_AGG_CFG, decode=_AGG_CFG)
_DP8_CFG = ReplicaParallelConfig(
    ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), replicas=1
)
_DISAGG_DP8_CFG = DisaggParallelConfig(prefill=_AGG_CFG, decode=_DP8_CFG)


def _config(**search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "deepseek-ai/DeepSeek-V3",
        "hardware_sku": "gb200",
        "backend": ["trtllm"],
        "deployment_mode": ["agg"],
        "gpu_budget": 16,
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"trace_path": TRACE},
    )


def _capabilities(*pairs):
    return RunnerCapabilities(supported_backend_topologies=tuple(pairs))


def test_branch_knobs_are_backend_only_and_mode_specific():
    search_space = _config().search_space

    agg = branch_knob_choices(search_space, "agg")
    disagg = branch_knob_choices(search_space, "disagg")

    assert set(agg) == {"agg_max_num_batched_tokens", "agg_max_num_seqs"}
    assert set(disagg) == {
        "prefill_max_num_batched_tokens",
        "prefill_max_num_seqs",
        "decode_max_num_batched_tokens",
        "decode_max_num_seqs",
    }
    assert not {"router_mode", "planner_scaling_policy", "num_g2_blocks"} & set(agg)


def test_actual_batch_and_context_candidates_replace_scheduler_aliases():
    search_space = _config(
        agg_batch_size_candidates=[16, 32],
        agg_context_tokens_candidates=[4096, 8192],
    ).search_space

    choices = branch_knob_choices(search_space, "agg")

    assert choices["agg_batch_size"] == [16, 32]
    assert choices["agg_context_tokens"] == [4096, 8192]
    assert "agg_max_num_seqs" not in choices
    assert "agg_max_num_batched_tokens" not in choices


def test_explicit_role_domains_and_replica_controls_reach_enumerator(monkeypatch):
    seen = {}

    def fake_parallel_configs(*args, **kwargs):
        seen.update(kwargs)
        return [_DISAGG_DP1_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["disagg"],
        backend=["vllm"],
        prefill_num_gpu_candidates=[4, 8],
        prefill_tp_candidates=[2, 4],
        prefill_pp_candidates=[1, 2],
        prefill_cp_candidates=[1, 4],
        prefill_num_workers_candidates=[1, 3],
        decode_num_gpu_candidates=[2],
        decode_tp_candidates=[2],
        decode_num_workers_candidates=[2],
        num_gpu_per_replica=[8, 16],
        max_gpu_per_replica=16,
        max_prefill_workers=3,
        max_decode_workers=4,
    )

    enumerate_branches(config)

    prefill = seen["prefill_candidates"]
    decode = seen["decode_candidates"]
    assert prefill.gpus_per_worker == (4, 8)
    assert prefill.tp == (2, 4)
    assert prefill.pp == (1, 2)
    assert prefill.cp == (1, 4)
    assert prefill.workers == (1, 3)
    assert decode.gpus_per_worker == (2,)
    assert decode.tp == (2,)
    assert decode.workers == (2,)
    assert seen["num_gpu_per_replica"] == (8, 16)
    assert seen["max_gpu_per_replica"] == 16
    assert seen["max_prefill_workers"] == 3
    assert seen["max_decode_workers"] == 4


def test_scheduler_limits_condition_exact_topology_domain_and_thorough_count(
    monkeypatch,
):
    seen = []

    def fake_parallel_configs(*args, **kwargs):
        del args
        limits = (kwargs["max_num_tokens"], kwargs["max_batch_size"])
        seen.append(limits)
        if limits == (4096, 1):
            return [_AGG_CFG]
        if limits == (8192, 2):
            return [_AGG_ALT_CFG]
        raise NoViableParallelConfig(f"scheduler limits {limits} do not fit")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        backend=["vllm"],
        agg_batch_size_candidates=[1, 2],
        agg_context_tokens_candidates=[4096, 8192],
    )

    (branch,) = enumerate_branches(config)
    sampler = ExhaustiveBranchSampler(branch)
    suggestions = sampler.suggest(10)

    assert seen == [(4096, 1), (8192, 1), (4096, 2), (8192, 2)]
    assert set(branch.parallel_configs) == {_AGG_CFG, _AGG_ALT_CFG}
    assert branch.enumeration_counts == {
        "considered": 4,
        "accepted": 2,
        "pruned": 2,
    }
    assert [diagnostic.as_dict() for diagnostic in branch.pruning_diagnostics] == [
        {
            "backend": "vllm",
            "role": None,
            "category": "no_parallel_config",
            "detail": (
                "2 scheduler point(s) pruned; first "
                "{'agg_batch_size': 1, 'agg_context_tokens': 8192}: "
                "scheduler limits (8192, 1) do not fit"
            ),
        }
    ]
    assert sampler.candidate_count == 2
    assert {
        (
            suggestion.parallel_config,
            suggestion.selection["agg_context_tokens"],
            suggestion.selection["agg_batch_size"],
        )
        for suggestion in suggestions
    } == {
        (_AGG_CFG, 4096, 1),
        (_AGG_ALT_CFG, 8192, 2),
    }


def test_large_infeasible_scheduler_limit_is_not_counted(monkeypatch):
    def fake_parallel_configs(*args, **kwargs):
        del args
        if kwargs["max_batch_size"] == 1:
            return [_AGG_CFG]
        raise NoViableParallelConfig("large scheduler batch does not fit")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        backend=["vllm"],
        agg_batch_size_candidates=[1, 512],
        agg_context_tokens_candidates=[8192],
    )

    (branch,) = enumerate_branches(config)
    sampler = ExhaustiveBranchSampler(branch)

    assert branch.enumeration_counts == {
        "considered": 2,
        "accepted": 1,
        "pruned": 1,
    }
    assert sampler.candidate_count == 1
    assert sampler.suggest(2)[0].selection["agg_batch_size"] == 1


def test_all_scheduler_points_pruned_retain_ordered_terminal_report(monkeypatch):
    def no_scheduler_point_fits(*args, **kwargs):
        del args
        if kwargs["backend"] == "vllm":
            raise NoPerfDatabase("vllm database missing")
        raise NoViableParallelConfig("sglang scheduler limits do not fit")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        no_scheduler_point_fits,
    )
    config = _config(
        backend=["vllm", "sglang"],
        agg_batch_size_candidates=[1, 2],
        agg_context_tokens_candidates=[4096, 8192],
    )

    with pytest.warns(UserWarning, match="no configured backend"), pytest.raises(
        NoViableParallelConfig
    ) as exc_info:
        enumerate_branches(config)

    assert exc_info.value.as_dict()["enumeration_reports"] == [
        {
            "deployment_mode": "agg",
            "counts": {"considered": 8, "accepted": 0, "pruned": 8},
            "pruning_diagnostics": [
                {
                    "backend": "vllm",
                    "role": None,
                    "category": "kv_capacity",
                    "detail": (
                        "4 scheduler point(s) pruned; first "
                        "{'agg_batch_size': 1, 'agg_context_tokens': 4096}: "
                        "vllm database missing"
                    ),
                },
                {
                    "backend": "sglang",
                    "role": None,
                    "category": "no_parallel_config",
                    "detail": (
                        "4 scheduler point(s) pruned; first "
                        "{'agg_batch_size': 1, 'agg_context_tokens': 4096}: "
                        "sglang scheduler limits do not fit"
                    ),
                },
            ],
        }
    ]


def test_enumerate_real_backend_space_honors_runner_topologies():
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=16,
    )
    capabilities = _capabilities(("trtllm", "agg"))

    with pytest.warns(UserWarning, match="runner-incompatible.*trtllm"):
        branches = enumerate_branches(config, runner_capabilities=capabilities)

    assert [branch.deployment_mode for branch in branches] == ["agg"]
    branch = branches[0]
    assert branch.knob_choices["backend"] == ["trtllm"]
    assert branch.parallel_configs
    assert all(config.total_gpus <= 16 for config in branch.parallel_configs)
    assert all(
        branch.supported_backends[config] == frozenset({"trtllm"})
        for config in branch.parallel_configs
    )
    assert "agg_max_num_seqs" in branch.knob_choices


def test_runner_incompatible_backend_is_removed_before_perf_lookup(monkeypatch):
    calls = []

    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
        **kwargs,
    ):
        del kwargs
        calls.append((deployment_mode, backend))
        return [_DISAGG_DP1_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["disagg"],
        backend=["trtllm", "vllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(
        config,
        runner_capabilities=_capabilities(("vllm", "disagg")),
    )

    assert calls
    assert set(calls) == {("disagg", "vllm")}
    assert branch.knob_choices["backend"] == ["vllm"]
    assert branch.supported_backends[_DISAGG_DP1_CFG] == frozenset({"vllm"})


def test_epd_runner_capability_is_checked_before_perf_lookup(monkeypatch):
    calls = []

    def fake_parallel_configs(*args, **kwargs):
        calls.append((args, kwargs))
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        fake_parallel_configs,
    )
    config = SmartSearchConfig.model_validate(
        {
            "search_space": {
                "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                "hardware_sku": "h200_sxm",
                "backend": ["vllm"],
                "deployment_mode": ["agg"],
                "gpu_budget": 8,
                "enable_epd": True,
            },
            "workload": {
                "trace_path": TRACE,
                "num_image_tokens": 256,
                "num_images_per_request": 1,
            },
        }
    )

    with (
        pytest.warns(UserWarning, match="runner-incompatible"),
        pytest.raises(NoViableParallelConfig),
    ):
        enumerate_branches(
            config,
            runner_capabilities=RunnerCapabilities(
                supported_backend_topologies=(("vllm", "agg"),),
            ),
        )
    assert calls == []

    (branch,) = enumerate_branches(
        config,
        runner_capabilities=RunnerCapabilities(
            supported_backend_topologies=(("vllm", "agg"),),
            supported_epd_backend_topologies=(("vllm", "agg"),),
        ),
    )
    assert calls
    assert branch.knob_choices["backend"] == ["vllm"]


def test_runner_prunes_disaggregated_attention_dp_before_sampling(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_DISAGG_DP1_CFG, _DISAGG_DP8_CFG],
    )
    config = _config(
        deployment_mode=["disagg"],
        backend=["vllm"],
        gpu_budget=16,
    )

    (branch,) = enumerate_branches(
        config,
        runner_capabilities=_capabilities(("vllm", "disagg")),
    )

    assert branch.parallel_configs == (_DISAGG_DP1_CFG,)
    assert _DISAGG_DP8_CFG not in branch.supported_backends


def test_runner_can_advertise_disaggregated_attention_dp(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_DISAGG_DP1_CFG, _DISAGG_DP8_CFG],
    )
    config = _config(
        deployment_mode=["disagg"],
        backend=["vllm"],
        gpu_budget=16,
    )
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "disagg"),),
        supports_disaggregated_attention_dp=True,
    )

    (branch,) = enumerate_branches(config, runner_capabilities=capabilities)

    assert branch.parallel_configs == (_DISAGG_DP1_CFG, _DISAGG_DP8_CFG)


def test_pinned_parallel_configs_replace_generated_menu():
    config = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 4, "replicas": 2}, {"tp": 8, "replicas": 1}],
    )

    (branch,) = enumerate_branches(config)

    assert {(item.shape.tp, item.replicas) for item in branch.parallel_configs} == {
        (4, 2),
        (8, 1),
    }
    assert all(item.total_gpus == 8 for item in branch.parallel_configs)


def test_illegal_pinned_parallel_config_is_rejected():
    config = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 3, "replicas": 1}],
    )

    with pytest.raises(NoViableParallelConfig):
        enumerate_branches(config)


def test_infeasible_mode_is_skipped_while_viable_mode_remains(monkeypatch):
    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
        **kwargs,
    ):
        del kwargs
        if deployment_mode == "disagg":
            raise NoViableParallelConfig("disagg does not fit")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=8,
    )

    with pytest.warns(UserWarning, match="disagg.*skipped"):
        branches = enumerate_branches(config)

    assert [branch.deployment_mode for branch in branches] == ["agg"]
    assert branches[0].supported_backends[_AGG_CFG] == frozenset({"trtllm"})


@pytest.mark.filterwarnings(
    "ignore:smart-sweep.*deployment_mode=.* skipped.*:UserWarning"
)
def test_all_modes_infeasible_raises(monkeypatch):
    def always_raise(*args, **kwargs):
        raise NoViableParallelConfig("nothing fits")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", always_raise
    )
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=1,
    )

    with pytest.raises(NoViableParallelConfig, match="no deployment_mode"):
        enumerate_branches(config)


def test_backend_without_perf_database_is_dropped(monkeypatch):
    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
        **kwargs,
    ):
        del kwargs
        if backend == "vllm":
            raise NoPerfDatabase("no vLLM perf database")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["vllm", "trtllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(config)

    assert branch.knob_choices["backend"] == ["trtllm"]
    assert branch.supported_backends[_AGG_CFG] == frozenset({"trtllm"})


def test_viable_backend_choices_preserve_user_order(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["trtllm", "vllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(config)

    assert branch.knob_choices["backend"] == ["trtllm", "vllm"]


def test_kv_load_range_becomes_continuous_branch_dimension(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = SmartSearchConfig(
        search_space={
            "model_name": "m",
            "hardware_sku": "h200_sxm",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.0, 1.0],
            "num_request_ratio": 10,
        },
        goal={"target": "pareto"},
    )

    (branch,) = enumerate_branches(config)

    assert branch.float_ranges == {"kv_load_ratio": (0.0, 1.0)}
    assert "kv_load_ratio" not in branch.knob_choices


def test_scalar_kv_load_is_pinned_in_branch_selection(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = SmartSearchConfig(
        search_space={
            "model_name": "m",
            "hardware_sku": "h200_sxm",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": 0.6,
            "num_request_ratio": 10,
        },
    )

    (branch,) = enumerate_branches(config)

    assert branch.float_ranges == {}
    assert branch.knob_choices["kv_load_ratio"] == [0.6]


def test_partial_illegal_pinned_config_raises(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["trtllm"],
        gpu_budget=8,
        parallel_configs=[{"tp": 1, "replicas": 1}, {"tp": 2, "replicas": 1}],
    )

    with pytest.raises(
        NoViableParallelConfig,
        match="legal/KV-feasible for no configured backend",
    ):
        enumerate_branches(config)
