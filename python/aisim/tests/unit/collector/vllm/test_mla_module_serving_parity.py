# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.32.0 serving parity of the MLA module collector, without vLLM or CUDA.

The fakes are strict: each ``vllm.*`` module carries only the names v0.32.0 has, so a
call into a name that release removed (0.30's ``Platform._find_non_ssm_backend``)
raises ``AttributeError`` here as it would in the container.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/vllm/collect_mla_module.py"


def _load_function(name, namespace=None):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def _install(monkeypatch, path, **attrs):
    module = types.ModuleType(path)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, path, module)


class _Platform:
    """v0.32.0 Platform surface used here: update_block_size_for_backend, no _find_non_ssm_backend."""

    def __init__(self, manager_block, *, sm=100, events=None):
        self.manager_block = manager_block
        self.sm = sm
        self.events = [] if events is None else events

    def update_block_size_for_backend(self, vllm_config):
        self.events.append("update")
        vllm_config.cache_config.block_size = self.manager_block

    def has_device_capability(self, capability):
        return self.sm >= capability


def test_page_is_serving_manager_block_narrowed_by_the_layer_backend(monkeypatch):
    events, calls = [], []
    platform = _Platform(64, events=events)
    _install(monkeypatch, "vllm.platforms", current_platform=platform)

    def select_common_block_size(manager_block, backends, specs):
        calls.append((manager_block, backends, specs))
        return 32

    _install(monkeypatch, "vllm.v1.worker.utils", select_common_block_size=select_common_block_size)
    config = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
    spec_seen = []
    layer = types.SimpleNamespace(
        get_attn_backend=lambda: "FLASHINFER_MLA",
        get_kv_cache_spec=lambda cfg: spec_seen.append(cfg.cache_config.block_size) or "spec",
    )

    page = _load_function("_serving_kv_block_size")(config, layer)

    assert page == 32
    assert events == ["update"]
    # The spec is built at the manager block serving chose, as serving builds the group spec.
    assert spec_seen == [64]
    assert calls == [(64, ["FLASHINFER_MLA"], ["spec"])]


def test_page_selection_needs_no_platform_name_removed_at_032(monkeypatch):
    platform = _Platform(64)
    assert not hasattr(platform, "_find_non_ssm_backend")
    _install(monkeypatch, "vllm.platforms", current_platform=platform)
    _install(monkeypatch, "vllm.v1.worker.utils", select_common_block_size=lambda block, backends, specs: block)
    config = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
    layer = types.SimpleNamespace(get_attn_backend=lambda: object(), get_kv_cache_spec=lambda cfg: None)
    assert _load_function("_serving_kv_block_size")(config, layer) == 64


class _CuteDsl:
    pass


class _Cutlass:
    pass


def _linear(kernel, holder="quant_method"):
    return types.SimpleNamespace(**{holder: types.SimpleNamespace(kernel=kernel)})


def _module(*linears):
    modules = [("", object())] + [(f"proj{i}", linear) for i, linear in enumerate(linears)]
    return types.SimpleNamespace(named_modules=lambda: modules)


def _autotuned_linears(monkeypatch, module, *, sm=120, flashinfer=True):
    _install(monkeypatch, "vllm.platforms", current_platform=_Platform(16, sm=sm))
    _install(monkeypatch, "vllm.utils.flashinfer", has_flashinfer=lambda: flashinfer)
    _install(
        monkeypatch,
        "vllm.model_executor.kernels.linear",
        FlashInferCuteDslNvFp4LinearKernel=_CuteDsl,
        FlashInferCutlassNvFp4LinearKernel=_Cutlass,
    )
    return _load_function("_serving_autotuned_linears")(module)


def test_cutlass_nvfp4_linears_are_the_ones_serving_autotunes(monkeypatch):
    q_b, o_proj, bf16 = _linear(_Cutlass()), _linear(_Cutlass(), holder="scheme"), _linear(None)
    assert _autotuned_linears(monkeypatch, _module(q_b, bf16, o_proj)) == [q_b, o_proj]


def test_a_cute_dsl_nvfp4_linear_skips_fp4_gemm_tuning_for_the_module(monkeypatch):
    module = _module(_linear(_Cutlass()), _linear(_CuteDsl()))
    assert _autotuned_linears(monkeypatch, module, sm=100) == []


@pytest.mark.parametrize(("sm", "flashinfer"), [(89, True), (120, False)])
def test_no_tuning_where_serving_runs_no_flashinfer_autotune(monkeypatch, sm, flashinfer):
    module = _module(_linear(_Cutlass()))
    assert _autotuned_linears(monkeypatch, module, sm=sm, flashinfer=flashinfer) == []


def test_modules_without_flashinfer_linears_have_nothing_to_tune(monkeypatch):
    assert _autotuned_linears(monkeypatch, _module(_linear(None), _linear(object()))) == []
