# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the fail-closed SGLang MoE lane guard.

The INT4/MXFP4 lanes run only on the SMs whose declared runner was traced to
what the pinned SGLang release serves; the collector's ``__compat__`` pin
already refuses any other release, so the guard keys on the SM alone. The
guard and its SM table are AST-extracted and run with the version and SM
helpers stubbed.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_moe.py"


def _load_guard(sm_version: int, installed_version: str = "0.5.23"):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_raise_if_unverified_moe_lane")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_VERIFIED_LANE_SMS" for target in node.targets)
        )
    ]
    loaded = {
        "_dist_version": lambda _name: installed_version,
        "get_sm_version": lambda: sm_version,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["_raise_if_unverified_moe_lane"]


@pytest.mark.parametrize(
    ("moe_type", "sm_version"),
    [
        # Kimi-K2.5 INT4: MXINT4 on flashinfer_trtllm (SM100/103), Marlin elsewhere.
        ("int4_wo", 89),
        ("int4_wo", 90),
        ("int4_wo", 100),
        ("int4_wo", 103),
        ("int4_wo", 120),
        # GPT-OSS W4A16: Triton on SM89/90, trtllm-gen with bf16 activations on SM100/103.
        ("w4a16_mxfp4", 89),
        ("w4a16_mxfp4", 90),
        ("w4a16_mxfp4", 100),
        ("w4a16_mxfp4", 103),
        # Kimi-K3 W4A16 on SM120: auto resolves Mxfp4MoEMethod to Triton.
        ("w4a16_mxfp4", 120),
        ("w4a8_mxfp4_mxfp8", 100),
        ("w4a8_mxfp4_mxfp8", 103),
        # SM120 GPT-OSS and DeepSeek-V4 FP4 experts: flashinfer_mxfp4 on the
        # CUTLASS MXFP8 x MXFP4 kernel.
        ("w4a8_mxfp4_mxfp8", 120),
    ],
)
def test_guard_accepts_traced_lanes(moe_type, sm_version):
    assert _load_guard(sm_version)(moe_type) == "0.5.23"


@pytest.mark.parametrize(
    ("moe_type", "sm_version"),
    [
        ("w4a8_mxfp4_mxfp8", 90),
    ],
)
def test_guard_rejects_untraced_lanes(moe_type, sm_version):
    with pytest.raises(RuntimeError, match=rf"{moe_type}.*installed: 0.5.23, SM{sm_version}"):
        _load_guard(sm_version)(moe_type)


@pytest.mark.parametrize("moe_type", ["bfloat16", "fp8_block", "nvfp4"])
@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 120])
def test_guard_never_fires_for_lanes_without_a_version_sensitive_runner(moe_type, sm_version):
    assert _load_guard(sm_version, "0.5.23+cu130")(moe_type) == "0.5.23+cu130"
