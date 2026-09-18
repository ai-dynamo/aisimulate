# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for residual-width MoE workspace accounting.

The source-isolated cases execute the real memory method, class constants and
vLLM-to-TRTLLM hook alias without requiring the compiled runtime. Only unrelated
speculation imports are replaced by inert types for these non-speculative
models. The final test also exercises the installed classes when available.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
BACKENDS = ROOT / "src" / "aisimulate_core" / "sdk" / "backends"
GIB = 1 << 30


def _source_backends():
    fields = {
        "ACTIVATION_COEFFICIENTS",
        "MOE_WORKSPACE_FAMILIES",
        "MIN_ACTIVATION_BYTES",
        "ACTIVATION_OVERHEAD_FRAC",
        "OTHERS_OVERHEAD_FRAC",
        "_moe_workspace_width",
    }
    methods = {"_moe_workspace_width", "_get_memory_usage"}
    namespace = {name: type(name, (), {}) for name in ("NullScheme", "SpecSchemeBase", "MTPScheme")}
    for filename, class_name in (
        ("base_backend.py", "BaseBackend"),
        ("trtllm_backend.py", "TRTLLMBackend"),
        ("vllm_backend.py", "VLLMBackend"),
    ):
        path = BACKENDS / filename
        tree = ast.parse(path.read_text(), filename=str(path))
        source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        selected = copy.deepcopy(source)
        selected.body = []
        for original in source.body:
            node = copy.deepcopy(original)
            if isinstance(node, ast.FunctionDef) and node.name in methods:
                for imported in [n for n in node.body if isinstance(n, ast.ImportFrom)]:
                    assert imported.module in {
                        "aisimulate_core.sdk.speculation",
                        "aisimulate_core.sdk.speculation.mtp",
                    }
                node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
                selected.body.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id in fields for t in targets):
                    selected.body.append(node)
        future = ast.parse("from __future__ import annotations").body
        module = ast.fix_missing_locations(ast.Module(body=[*future, selected], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def backends():
    return _source_backends()


def _model(*, tp=4, dp=1, ep=1, hidden_size=7168, heads=128):
    return SimpleNamespace(
        context_ops=[SimpleNamespace(get_weights=lambda: 2 * GIB)],
        model_family="DEEPSEEKV4",
        _num_heads=heads,
        _head_size=512,
        _hidden_size=hidden_size,
        _num_experts=384,
        _topk=6,
        config=SimpleNamespace(pp_size=1, tp_size=tp, attention_dp_size=dp, moe_ep_size=ep, nextn=0),
        get_kvcache_bytes_per_sequence=lambda _seq: 1024,
        _cp_kv_memory_divisor=lambda: 1,
    )


def _activation_gib(backend, model, tokens):
    database = SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: GIB, 4: GIB}, "other_mem": 3 * GIB}})
    result = backend._get_memory_usage(
        model,
        database,
        batch_size=1,
        beam_width=1,
        isl=1,
        osl=1,
        num_tokens=tokens,
        mtp_activation_scaling=False,
    )
    assert result["weights"] == 2
    assert result["nccl"] == 1
    assert result["others"] == 3
    assert result["kvcache"] == 1024 / GIB
    return result["activations"]


@pytest.mark.parametrize("backend_name", ["TRTLLMBackend", "VLLMBackend"])
@pytest.mark.parametrize("tokens", [2048, 8192])
@pytest.mark.parametrize("tp,dp,ep", [(4, 1, 1), (1, 8, 8), (1, 4, 1)])
def test_dsv4_moe_workspace_uses_residual_width(backends, backend_name, tokens, tp, dp, ep):
    backend = backends[backend_name]()
    model = _model(tp=tp, dp=dp, ep=ep)
    attention_width = model._num_heads * model._head_size
    coefficient = backend.ACTIVATION_COEFFICIENTS[model.model_family][tp]
    base_activation = 2 * tokens * attention_width * coefficient
    workspace = tokens * model._hidden_size * dp * model._num_experts * model._topk / ep / 128 * 4
    # Keep the expanded attention term unchanged; only the MoE buffer has h=7168.
    assert _activation_gib(backend, model, tokens) == pytest.approx((base_activation + workspace) / GIB)


@pytest.mark.parametrize("backend_name", ["TRTLLMBackend", "VLLMBackend"])
@pytest.mark.parametrize("hidden_size,heads", [(7168, 128), (4096, 64)])
def test_dsv4_hook_uses_model_geometry_not_a_pro_specific_constant(backends, backend_name, hidden_size, heads):
    model = _model(hidden_size=hidden_size, heads=heads)
    assert backends[backend_name]()._moe_workspace_width(model, "DEEPSEEKV4", heads * 512) == hidden_size


@pytest.mark.parametrize("backend_name", ["TRTLLMBackend", "VLLMBackend"])
def test_legacy_deepseek_and_existing_hybrid_behavior_are_preserved(backends, backend_name):
    backend = backends[backend_name]()
    model = _model()
    assert backend._moe_workspace_width(model, "DEEPSEEK", 65536) == 65536
    for family in ("GEMMA4MIX", "STEP3P7"):
        assert backend._moe_workspace_width(model, family, 65536) == 7168
    assert backend._moe_workspace_width(SimpleNamespace(), "DEEPSEEKV4", 65536) == 65536


def test_installed_backends_match_source_isolated_budget():
    vllm = pytest.importorskip("aisimulate_core.sdk.backends.vllm_backend")
    trtllm = pytest.importorskip("aisimulate_core.sdk.backends.trtllm_backend")
    for backend in (vllm.VLLMBackend(), trtllm.TRTLLMBackend()):
        assert _activation_gib(backend, _model(), 8192) == pytest.approx(13.9375)
        assert _activation_gib(backend, _model(tp=1, dp=8, ep=8), 8192) == pytest.approx(25.9375)
