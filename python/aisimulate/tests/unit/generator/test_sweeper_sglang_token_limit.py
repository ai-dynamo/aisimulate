# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.generator.api import generate_from_request
from aiconfigurator.generator.request import ModelFacts, from_sweeper_candidate

from .test_sweeper_request import _agg_candidate, _cli_flag_value, _disagg_candidate


def test_agg_sglang_candidate_renders_evaluated_prefill_token_limit():
    request = from_sweeper_candidate(
        _agg_candidate(backend="sglang", backend_version="0.5.18"),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:0.5.18"}},
    )

    artifacts = generate_from_request(request)

    assert request.topology.roles["agg"].extra["max_prefill_tokens"] == 8192
    assert _cli_flag_value(artifacts["cli_args_agg"], "--max-prefill-tokens") == "8192"


def test_disagg_sglang_candidate_renders_each_role_prefill_token_limit():
    adapters = {
        "dynamo.router": {
            "mode": "kv",
            "overlap_score_credit": 0.75,
            "router_temperature": 0.2,
        },
        "dynamo.planner": {
            "scaling_policy": "throughput",
            "enable_throughput_scaling": True,
            "enable_load_scaling": False,
            "environment": "kubernetes",
            "mode": "disagg",
        },
    }
    request = from_sweeper_candidate(
        _disagg_candidate(
            backend="sglang",
            backend_version="0.5.18",
            adapters=adapters,
        ),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:0.5.18"}},
    )

    artifacts = generate_from_request(request)

    assert _cli_flag_value(artifacts["cli_args_prefill"], "--max-prefill-tokens") == "16384"
    assert _cli_flag_value(artifacts["cli_args_decode"], "--max-prefill-tokens") == "8192"
