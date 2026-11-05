# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check synthetic native capacity without importing the GPU-only framework."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector"


@pytest.mark.parametrize(
    "input_len,output_len,is_context,capacity",
    [
        (128, 1, True, 163840),
        (128, 1, False, 163840),
        (1048576, 1, True, 1048577),
        (1048575, 1, False, 1048577),
        (1048575, 4, False, 1048583),
    ],
)
def test_native_run_sizes_rope_to_cache_capacity_without_changing_yarn(input_len, output_len, is_context, capacity):
    path = COLLECTOR / "trtllm/collect_mla.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in {"Scenario", "RopeConfig", "run_mla"}
    ]
    calls = []
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "torch": SimpleNamespace(dtype=object, bfloat16="bfloat16"),
        "get_sm_version": lambda: 103,
        "_run_attn_for_backend": lambda *args: calls.append(args),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["run_mla"](input_len, 1, output_len, "bf16", 128, 8, 8, 32, 0, 1, is_context, perf_filename="unused")
    assert len(calls) == 1
    config = calls[0][8]
    assert config.max_position_embeddings == capacity
    assert config.rope_scaling == {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    }
    assert config.rope_theta == 10000.0
    assert config.qk_rope_head_dim == 64
