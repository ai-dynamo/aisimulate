# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All-reduce collector setup for the GLM-5.3-Flash runtimes (vLLM 0.30.0, SGLang 0.5.20).

Static (ast) checks: the collector needs a framework plus GPUs to import.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_COLLECTOR = Path(__file__).resolve().parents[3] / "collector" / "network" / "collect_all_reduce.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_COLLECTOR.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _load_sglang_server_args_block(server_args_cls):
    """Execute benchmark_sglang_allreduce's server-args preamble against a fake SGLang."""
    function = _function("benchmark_sglang_allreduce")
    published = []
    stop = RuntimeError("stop after publishing server args")

    def setup(*_args, **_kwargs):
        raise stop

    namespace = {
        "setup_sglang_distributed": setup,
        "torch": None,
    }
    fake_module = type("M", (), {})()
    fake_module.ServerArgs = server_args_cls
    fake_module.set_global_server_args_for_scheduler = published.append

    import sys

    saved = sys.modules.get("sglang.srt.server_args")
    sys.modules["sglang.srt.server_args"] = fake_module
    try:
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(_COLLECTOR), "exec"), namespace)
        with pytest.raises(RuntimeError, match="stop after publishing"):
            namespace["benchmark_sglang_allreduce"]("bfloat16", "128,256,2", 2, 0, False, "x.txt")
    finally:
        if saved is None:
            sys.modules.pop("sglang.srt.server_args", None)
        else:
            sys.modules["sglang.srt.server_args"] = saved
    return published


def test_sglang_0_5_20_publishes_real_server_args():
    class ServerArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def resolve_once(self):
            return None

    (published,) = _load_sglang_server_args_block(ServerArgs)
    assert isinstance(published, ServerArgs)
    assert published.kwargs == {"model_path": "dummy", "enable_symm_mem": False}


def test_older_sglang_keeps_the_mock_server_args():
    class ServerArgs:  # no resolve_once: pre-0.5.20 publishing accepts a mock
        pass

    (published,) = _load_sglang_server_args_block(ServerArgs)
    assert type(published).__name__ == "MockServerArgs"
    assert published.enable_symm_mem is False


def test_vllm_graph_mode_warms_each_shape_eagerly_before_capture():
    source = ast.get_source_segment(_COLLECTOR.read_text(encoding="utf-8"), _function("benchmark_vllm_allreduce"))
    warm = source.index('vllm_mods["tensor_model_parallel_all_reduce"](warm)')
    capture = source.index('vllm_mods["graph_capture"](')
    assert warm < capture


def test_vllm_graph_capture_uses_the_registered_serving_pool():
    source = ast.get_source_segment(_COLLECTOR.read_text(encoding="utf-8"), _function("benchmark_vllm_allreduce"))
    pool = source.index("graph_pool = _vllm_graph_pool()")
    capture = source.index("torch.cuda.graph(graph, pool=graph_pool")
    assert pool < capture
    helper = ast.get_source_segment(_COLLECTOR.read_text(encoding="utf-8"), _function("_vllm_graph_pool"))
    assert "set_graph_pool_id(pool)" in helper
