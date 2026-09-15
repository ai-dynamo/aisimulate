# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import math
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def parser():
    source = Path(__file__).resolve().parents[3] / "collector/network/collect_nccl.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_nccl_latency")
    namespace = {"math": math, "re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[fn.name]


def test_nccl_parser_uses_exact_size_runtime_and_microseconds():
    text = "# informational time string\n512 256 half sum -1 7.5 1 1 0 7.2 1 1 0\n"
    assert parser()(text, "NCCL version 2.28.9+cuda13.0", 512, "2.28.9") == 0.0075


@pytest.mark.parametrize(
    "version,size,latency",
    [("2.27.3", 512, "7.5"), ("", 512, "7.5"), ("2.28.9", 1024, "7.5"), ("2.28.9", 512, "nan"), ("2.28.9", 512, "0")],
)
def test_nccl_parser_rejects_wrong_or_unproven_measurements(version, size, latency):
    text = f"512 256 half sum -1 {latency} 1 1 0 7.2 1 1 0\n"
    with pytest.raises(RuntimeError):
        parser()(text, f"NCCL version {version}", size, "2.28.9")
