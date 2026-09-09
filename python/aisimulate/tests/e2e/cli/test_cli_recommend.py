# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from ai-dynamo/AIConfigurator at commit
# 77fd0773407b3683d8a671fe24a30a7110651b64; modified for the unified AISimulate package.

"""End-to-end tests for recommendation GPU-budget escalation."""

import pytest

from aiconfigurator.cli.api import cli_recommend

pytestmark = [pytest.mark.e2e, pytest.mark.build]


def test_recommend_multi_node_moe_returns_results():
    """A large MoE model escalates beyond a single B200 node."""
    result = cli_recommend(
        model_path="moonshotai/Kimi-K3",
        system="b200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        target_concurrency=16,
        database_mode="SILICON",
    )

    assert result.chosen_exp is not None
    best = result.best_configs.get(result.chosen_exp)
    assert best is not None and not best.empty, "Expected at least one recommended config"

    top = best.iloc[0]
    assert top["num_total_gpus"] > 8, f"Expected multi-node config (>8 GPUs), got {top['num_total_gpus']}"
    assert top["ttft"] > 0
    assert top["tpot"] > 0


def test_recommend_single_node_dense_model():
    """A small dense model stays within one H200 node."""
    result = cli_recommend(
        model_path="meta-llama/Meta-Llama-3.1-8B",
        system="h200_sxm",
        backend="vllm",
        isl=4000,
        osl=1000,
        target_concurrency=32,
        database_mode="HYBRID",
    )

    assert result.chosen_exp is not None
    best = result.best_configs.get(result.chosen_exp)
    assert best is not None and not best.empty
    assert best.iloc[0]["num_total_gpus"] <= 8
