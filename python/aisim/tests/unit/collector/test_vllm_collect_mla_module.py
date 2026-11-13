# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-capacity refusals of the vLLM MLA/DSA module collector.

The module op plans do not depend on the device (collector/op_plans.py), so
run_mla_module refuses a generation point whose KV cache cannot fit the device
before allocating it, with a classified FIXME(kernel-limit) reason. These tests
load the real function bodies from source (vLLM itself is not importable in CI)
and pin the two properties that matter:

- a point whose KV lower bound exceeds the device budget is refused;
- on >= 90 GB devices (H20/B200/RTX PRO 6000) no point of the decode grid is
  refused, i.e. the refusal cannot change SM90/SM100/SM120 behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MLA_MODULE_SOURCE = ROOT / "collector/vllm/collect_mla_module.py"


def _load_function(path: Path, name: str, namespace: dict):
    """Compile the real function body without importing unavailable vLLM."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _load_assignment(path: Path, name: str):
    """Read a module-level constant from source."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    expr = ast.fix_missing_locations(ast.Expression(node.value))
                    return eval(compile(expr, str(path), "eval"), {})  # our own source
    raise AssertionError(f"{name} not found in {path}")


L40S_TOTAL_MEMORY = 47_669_248_000  # 44.39 GiB usable, as reported by torch
H20_TOTAL_MEMORY = 102_600_000_000  # ~96 GB class devices (H20 / RTX PRO 6000)

GENERATION_BATCH_SIZES = [256, 512, 1024]
GENERATION_SEQUENCE_LENGTHS = [16384, 32768, 65536, 131072]
GENERATION_MAX_TOKENS = 2**25


def _build_kv_does_not_fit():
    namespace = {
        "_MLA_KV_ENTRY_ELEMS": _load_assignment(MLA_MODULE_SOURCE, "_MLA_KV_ENTRY_ELEMS"),
        "_MEMORY_BUDGET_SAFETY_FACTOR": _load_assignment(MLA_MODULE_SOURCE, "_MEMORY_BUDGET_SAFETY_FACTOR"),
    }
    _load_function(MLA_MODULE_SOURCE, "_generation_kv_footprint_bytes", namespace)
    return _load_function(MLA_MODULE_SOURCE, "_kv_does_not_fit", namespace)


def _refused(total_memory, kv_cache_dtype):
    does_not_fit = _build_kv_does_not_fit()
    return {
        (s, b): reason
        for b in GENERATION_BATCH_SIZES
        for s in GENERATION_SEQUENCE_LENGTHS
        if b * s <= GENERATION_MAX_TOKENS and (reason := does_not_fit(b * s, kv_cache_dtype, total_memory)) is not None
    }


def test_points_the_device_cannot_hold_are_refused_with_their_reason():
    refused = _refused(L40S_TOTAL_MEMORY, "bfloat16")
    # 2 x 33.5M tokens x 576 x 2B = 77.3 GB cannot fit on a 46 GB device...
    assert set(refused) == {(32768, 1024), (65536, 512), (131072, 256)}
    # ...while the 16.7M-token points (2 x 19.3 GB) still run.
    assert all(
        reason.startswith("FIXME(kernel-limit): does not fit: the KV cache needs 72.0 GiB")
        for reason in refused.values()
    )


def test_no_point_is_refused_on_90gb_class_devices():
    assert _refused(H20_TOTAL_MEMORY, "bfloat16") == {}


def test_fp8_kv_halves_the_bound():
    # 2 x 33.5M x 576 x 1B = 38.7 GB fits the 42.9 GB budget.
    assert _refused(L40S_TOTAL_MEMORY, "fp8") == {}
