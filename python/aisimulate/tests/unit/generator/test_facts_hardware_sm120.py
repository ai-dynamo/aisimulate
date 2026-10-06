# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTX PRO (sm120) hardware facts profile. Without it resolve_facts() raised
"Unknown hardware profile 'rtx_pro_6000_server'", the pipeline swallowed the
error and every sm120 render lost its model defaults (op-probe harness,
2026-10-01: V4.1's block-size 64 never reached sm120 engine args)."""

import pytest

from aisimulate.generator.facts.apply import apply_model_default_args
from aisimulate.generator.facts.request_resolution import hardware_key_for_system
from aisimulate.generator.facts.resolve import _FACTS_DIR, load_backend_version_matrix, resolve_facts

pytestmark = pytest.mark.unit


def _latest_dynamo() -> str:
    matrix = load_backend_version_matrix(str(_FACTS_DIR / "runtimes" / "dynamo.yaml"))
    return next(v for v, backends in matrix.items() if "vllm" in backends)


def test_rtx_pro_6000_server_maps_to_the_sm120_profile():
    assert hardware_key_for_system("rtx_pro_6000_server") == "sm120"
    # unknown systems still fall through to the raw name (clear error downstream)
    assert hardware_key_for_system("not_a_system") == "not_a_system"


def test_sm120_profile_resolves_and_carries_no_wideep_moe_backend():
    facts = resolve_facts(
        model_profile_id="deepseek-v4.1",
        hardware="sm120",
        transport="nvlink",
        dynamo_version=_latest_dynamo(),
        backend="vllm",
    )
    assert facts.hardware["arch"] == "x86_64"
    assert facts.hardware["node_selector"]["nvidia.com/gpu.product"].startswith("NVIDIA-RTX-PRO-6000")
    # PCIe Blackwell: no wide-EP MoE backend is asserted for any backend
    assert "moe_backend" not in facts.hardware


def test_model_defaults_reach_sm120_engine_args():
    facts = resolve_facts(
        model_profile_id="deepseek-v4.1",
        hardware="sm120",
        transport="nvlink",
        dynamo_version=_latest_dynamo(),
        backend="vllm",
    )
    tokens: list[str] = []
    apply_model_default_args(tokens, facts.model, backend="vllm", system="sm120", role="agg", variant=None)
    assert tokens[tokens.index("--block-size") + 1] == "64"
    assert "--trust-remote-code" in tokens
