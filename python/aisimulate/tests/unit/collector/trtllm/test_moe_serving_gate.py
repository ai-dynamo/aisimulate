# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free: the TRT-LLM MoE collector routes through the architecture's serving gate with a zeroed bias."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECT_MOE = Path(__file__).resolve().parents[4] / "collector" / "trtllm" / "collect_moe.py"


class _Bias:
    zeroed = False

    def zero_(self):
        self.zeroed = True


def _gate(kind):
    class Gate:
        def __init__(self, *args, **kwargs):
            self.kind = kind
            self.kwargs = kwargs
            self.e_score_correction_bias = _Bias()

    return Gate


@pytest.fixture
def serving_gate(monkeypatch):
    tree = ast.parse(COLLECT_MOE.read_text())
    helpers = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_zero_routing_bias", "_serving_moe_gate")
    ]
    namespace = {"torch": SimpleNamespace(no_grad=nullcontext), "DeepseekV3Gate": _gate("v3")}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(COLLECT_MOE), "exec"), namespace)
    for name in ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attr, kind in (
        ("modeling_deepseekv4", "DeepseekV4Gate", "v4"),
        ("modeling_minimaxm3", "MiniMaxM3Gate", "m3"),
    ):
        module = ModuleType(f"tensorrt_llm._torch.models.{name}")
        setattr(module, attr, _gate(kind))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return namespace["_serving_moe_gate"]


@pytest.mark.parametrize(
    ("architecture", "kind"),
    [
        ("DeepseekV3ForCausalLM", "v3"),
        ("GlmMoeDsaForCausalLM", "v3"),
        ("DeepseekV4ForCausalLM", "v4"),
        ("MiniMaxM3ForCausalLM", "m3"),
    ],
)
def test_gate_matches_the_architecture_and_its_bias_is_zeroed(serving_gate, architecture, kind):
    declared = SimpleNamespace(
        architecture=architecture,
        sglang_moe_num_expert_group=8,
        sglang_moe_topk_group=4,
        sglang_moe_routed_scaling_factor=2.5,
    )
    gate = serving_gate(declared, 7168, 256, 8, "bf16", "trtllm")
    assert gate.kind == kind
    assert gate.e_score_correction_bias.zeroed
    if kind == "v3":
        # serving compares the resolved backend upper-case: 'TRTLLM' keeps a bf16 bias
        assert gate.kwargs["moe_backend"] == "TRTLLM"
    if kind == "v4":
        assert gate.kwargs["is_hashed"] is False
