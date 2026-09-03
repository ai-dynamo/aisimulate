# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the fail-closed SGLang MoE lane version guard.

The version-sensitive INT4/MXFP4 dispatch lanes are source-verified for the
0.5.14 and 0.5.17 patch series. Other series remain rejected, while the
Qwen3.8-Max lanes collected by this change are not version-gated.

AST-extracts the function the same way ``test_collect_gdn_contract.py``'s
``TestResolveFlashinferGdnDecode`` and this directory's other collector
function tests do; ``pkg_resources.get_distribution`` is stubbed (no
sys.modules injection needed here -- the guard reads the installed version
through ``pkg_resources``, not through an import-time capability probe like
the runner-backend pin), and the real ``collector.version_resolver.
_check_compat`` is used unmocked so the test exercises the actual version
grammar, not a re-implementation of it.
"""

import ast
import types
from pathlib import Path

import pytest
from collector.version_resolver import _check_compat

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_moe.py"

UNVERIFIED_LANES = ["int4_wo", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"]
QWEN38MAX_LANES = ["bfloat16", "fp8_block", "nvfp4"]


def _load_guard(installed_version: str):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_raise_if_unverified_moe_lane"
    )
    fake_distribution = types.SimpleNamespace(version=installed_version)
    fake_pkg_resources = types.SimpleNamespace(get_distribution=lambda _name: fake_distribution)
    loaded = {"pkg_resources": fake_pkg_resources, "_check_compat": _check_compat}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["_raise_if_unverified_moe_lane"]


@pytest.mark.parametrize(
    "installed_version",
    ["0.5.14", "0.5.14.post1", "0.5.14+cu128", "0.5.17", "0.5.17.post1", "0.5.17+cu128"],
)
@pytest.mark.parametrize("moe_type", UNVERIFIED_LANES)
def test_guard_accepts_verified_patch_series(moe_type, installed_version):
    guard = _load_guard(installed_version)

    assert guard(moe_type) == installed_version


@pytest.mark.parametrize("installed_version", ["0.5.13", "0.5.15", "0.5.16", "0.5.18"])
@pytest.mark.parametrize("moe_type", UNVERIFIED_LANES)
def test_guard_rejects_unverified_series(moe_type, installed_version):
    guard = _load_guard(installed_version)

    with pytest.raises(RuntimeError, match=rf"{moe_type}.*installed: {installed_version}"):
        guard(moe_type)


@pytest.mark.parametrize("installed_version", ["0.5.14", "0.5.17"])
@pytest.mark.parametrize("moe_type", QWEN38MAX_LANES)
def test_guard_never_fires_for_qwen38max_collected_lanes(moe_type, installed_version):
    """The three lanes this bump actually verified must never be gated by
    this guard, at either pinned version -- a bug that widened the guard's
    scope would silently break Qwen3.8-Max collection."""
    guard = _load_guard(installed_version)

    assert guard(moe_type) == installed_version
