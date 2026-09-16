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


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_nccl_failure_always_stops_sampling_and_preserves_process_error(cleanup_fails):
    import os
    import subprocess
    import sys
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[3] / "collector/network/collect_nccl.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "nccl_benchmark")
    calls = []
    failure = subprocess.CalledProcessError(7, ["nccl-test"])

    class Monitor:
        def __init__(self, **kwargs):
            pass

        def _init_handle(self):
            return True

        def start_sampling(self):
            calls.append("start")

        def stop_sampling(self):
            calls.append("stop")
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    def run(*args, **kwargs):
        assert kwargs["check"] is True
        raise failure

    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(nccl=SimpleNamespace(version=lambda: (2, 28, 9)))),
        "PowerMonitor": Monitor,
        "subprocess": SimpleNamespace(run=run),
        "os": os,
        "sys": sys,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(subprocess.CalledProcessError) as result:
        namespace[fn.name]("half", test_range="512,1024,2", measure_power=True)
    assert result.value is failure
    assert calls == ["start", "stop"]
