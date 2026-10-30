# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No collector may time a kernel eagerly behind the CUDA-graph fallback.

``benchmark_with_power(allow_graph_fail=True)`` replaces a failed capture with
the Python launch loop and ships that floor as kernel latency; no perf table
records the helper's ``used_cuda_graph`` flag, and layer_permissions.md allows
a measurement-method degradation only when it is recorded in the output row.
The sglang trtllm_mla prefill (B1) and the trtllm GDN lanes were closed on
2026-10-11, the last four sites — trtllm mamba2 and moe, wideep moe_compute,
network trtllm alltoall — on 2026-10-12. This test keeps the count at zero for
every file under ``collector/``: any ``allow_graph_fail=`` other than the literal
``False`` (a variable could carry ``True``) is an offender. The one sound exception
is a wrapper forwarding its own keyword-only ``allow_graph_fail=False`` parameter:
every call of that wrapper passes the keyword (or the False default) and is scanned
by the same walk."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COLLECTOR_ROOT = Path(__file__).resolve().parents[3] / "collector"


def _forwards_its_false_default(function, value):
    """``allow_graph_fail=allow_graph_fail`` inside a function whose keyword-only parameter defaults to False."""
    if function is None or not (isinstance(value, ast.Name) and value.id == "allow_graph_fail"):
        return False
    for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True):
        if arg.arg == "allow_graph_fail":
            return isinstance(default, ast.Constant) and default.value is False
    return False


def test_no_collector_passes_allow_graph_fail_other_than_literal_false():
    offenders = []
    for path in sorted(COLLECTOR_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        enclosing = {}  # node -> innermost enclosing function (ast.walk visits outer functions first)
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(function):
                    enclosing[child] = function
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "allow_graph_fail":
                    continue
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    continue
                if _forwards_its_false_default(enclosing.get(node), keyword.value):
                    continue
                offenders.append(f"{path.relative_to(COLLECTOR_ROOT)}:{node.lineno}")
    assert offenders == [], f"eager-timed rows would ship from: {offenders}"
