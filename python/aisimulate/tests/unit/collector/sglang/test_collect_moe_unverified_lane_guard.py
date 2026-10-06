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


def _load_guard(installed_version: str, sm_version: int = 90):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_raise_if_unverified_moe_lane"
    )
    fake_distribution = types.SimpleNamespace(version=installed_version)
    fake_pkg_resources = types.SimpleNamespace(get_distribution=lambda _name: fake_distribution)
    loaded = {
        "pkg_resources": fake_pkg_resources,
        "_check_compat": _check_compat,
        # the guard reads the runtime through the module helpers; stub both
        "_dist_version": lambda _name: installed_version,
        "get_sm_version": lambda: sm_version,
    }
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


@pytest.mark.parametrize("installed_version", ["0.5.21", "0.5.21.post1", "0.5.21+cu130"])
@pytest.mark.parametrize("sm_version", [89, 90])
@pytest.mark.parametrize("moe_type", ["int4_wo", "w4a16_mxfp4"])
def test_guard_accepts_0521_hopper_lanes_reverified_2026_10_04(moe_type, installed_version, sm_version):
    """0.5.21 Marlin (int4_wo) and Triton (w4a16_mxfp4) dispatch re-verified on SM89/90 only."""
    guard = _load_guard(installed_version, sm_version)
    assert guard(moe_type) == installed_version


@pytest.mark.parametrize("moe_type", ["int4_wo", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"])
@pytest.mark.parametrize("sm_version", [100, 103])
def test_guard_accepts_0521_blackwell_mxfp4_lanes(moe_type, sm_version):
    """SM100/103 MXFP4 lanes re-verified on 0.5.21 from the B200 identity records (6400345f);
    int4_wo re-opened 2026-10-06 from the same records (Kimi-K2.5 -> FLASHINFER_TRTLLM / trtllm_gen_moe)."""
    guard = _load_guard("0.5.21", sm_version)
    assert guard(moe_type) == "0.5.21"


@pytest.mark.parametrize(
    "moe_type, sm_version",
    [("w4a8_mxfp4_mxfp8", 90), ("int4_wo", 120), ("w4a16_mxfp4", 120)],
)
def test_guard_keeps_0521_unverified_lanes_closed(moe_type, sm_version):
    """Every SM120 weight-only lane and the DSV4 FP4 lane on Hopper were not re-verified."""
    guard = _load_guard("0.5.21", sm_version)
    with pytest.raises(RuntimeError, match=rf"{moe_type}.*installed: 0.5.21, SM{sm_version}"):
        guard(moe_type)
