# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual worker scope and absolute-position construction without CUDA."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3] / "collector/vllm"


def load_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "filename,worker,target",
    [
        ("collect_mla_module.py", "run_mla_module_worker", "run_mla_module"),
        ("collect_msa_module.py", "run_msa_module_worker", "run_msa_module"),
    ],
)
@pytest.mark.parametrize("raises", [False, True])
def test_inference_mode_spans_entire_callback_and_restores_on_error(filename, worker, target, raises):
    state = {"inference": False}

    @contextmanager
    def inference_mode():
        assert not state["inference"]
        state["inference"] = True
        try:
            yield
        finally:
            state["inference"] = False

    def execute(**kwargs):
        assert state["inference"]
        assert kwargs["prefix_len"] == 128
        if raises:
            raise RuntimeError("runtime failure retained")
        return "measured"

    fn = load_function(filename, worker, {"torch": SimpleNamespace(inference_mode=inference_mode), target: execute})
    args = [16, 2, 64, "fp8", "bfloat16", "fp8_block", "fixture/model"]
    if "mla" in filename:
        args.append("mla")
    if raises:
        with pytest.raises(RuntimeError, match="runtime failure retained"):
            fn(*args, prefix_len=128, perf_filename="context_perf.txt")
    else:
        assert fn(*args, prefix_len=128, perf_filename="context_perf.txt") == "measured"
    assert not state["inference"]


@pytest.mark.parametrize(
    "batch,length,context,prefix,expected",
    [
        (2, 3, True, 0, [0, 1, 2, 0, 1, 2]),
        (2, 3, True, 128, [128, 129, 130, 128, 129, 130]),
        (3, 2048, False, 0, [2048, 2048, 2048]),
    ],
)
def test_msa_positions_follow_cached_plus_query_coordinate(batch, length, context, prefix, expected):
    fn = load_function("collect_msa_module.py", "_msa_query_positions", {})
    assert fn(batch, length, context, prefix) == expected


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.24.0", (4, 8, 16)),
        ("0.25.0", (8, 4, 16)),
        ("0.25.0+cu130", (8, 4, 16)),
    ],
)
def test_msa_topk_layout_preserves_024_and_uses_025_token_major(version, expected):
    fn = load_function("collect_msa_module.py", "_msa_topk_buffer_shape", {})
    assert fn(4, 5, 16, version) == expected
