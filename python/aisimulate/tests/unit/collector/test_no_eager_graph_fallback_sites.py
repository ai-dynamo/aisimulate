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
every file under ``collector/``."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COLLECTOR_ROOT = Path(__file__).resolve().parents[3] / "collector"

# Former fallback sites: number of benchmark_with_power calls each still makes.
FORMER_SITES = {
    "network/slurm/collect_trtllm_alltoall.py": 1,
    "trtllm/collect_mamba2.py": 8,
    "trtllm/collect_moe.py": 1,
    "vllm/collect_moe_xpu.py": 1,
    "wideep/trtllm/collect_moe_compute.py": 1,
}


def _benchmark_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "benchmark_with_power"
    ]


def test_no_collector_passes_allow_graph_fail_true():
    offenders = []
    for path in sorted(COLLECTOR_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "allow_graph_fail"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(f"{path.relative_to(COLLECTOR_ROOT)}:{node.lineno}")
    assert offenders == [], f"eager-timed rows would ship from: {offenders}"


@pytest.mark.parametrize(("relative_path", "expected_calls"), sorted(FORMER_SITES.items()))
def test_former_fallback_sites_time_under_cuda_graphs_only(relative_path, expected_calls):
    tree = ast.parse((COLLECTOR_ROOT / relative_path).read_text(encoding="utf-8"))
    calls = _benchmark_calls(tree)
    assert len(calls) == expected_calls
    for call in calls:
        assert all(keyword.arg != "allow_graph_fail" for keyword in call.keywords), relative_path
