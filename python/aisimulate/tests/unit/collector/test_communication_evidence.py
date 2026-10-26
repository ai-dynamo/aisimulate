# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
NETWORK = Path(__file__).resolve().parents[3] / "collector/network"


def load(name):
    source = NETWORK / "collect_all_reduce.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"sys": sys, "contextmanager": contextmanager, "Path": Path, "json": json, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_dispatch_records_successful_path_and_restores_profiler():
    namespace = {"__name__": "vllm.distributed.device_communicators.pynccl"}
    exec("def all_reduce(value):\n return value\n", namespace)
    events = set()
    previous = sys.getprofile()
    with pytest.raises(RuntimeError, match="probe failed"), load("_observe_vllm_dispatch")(events):
        namespace["all_reduce"](None)
        assert not events
        namespace["all_reduce"](object())
        raise RuntimeError("probe failed")
    assert events == {"vllm.distributed.device_communicators.pynccl.all_reduce"}
    assert sys.getprofile() is previous


@pytest.mark.parametrize(
    "value,events,error", [(4, {"actual.path"}, None), (3, {"actual.path"}, "numerical"), (4, set(), "observed")]
)
def test_evidence_is_saved_before_gate_failure(tmp_path, value, events, error):
    scalar = SimpleNamespace(item=lambda: value)
    output = SimpleNamespace(amin=lambda: scalar, amax=lambda: scalar, shape=(1, 128))
    arguments = (str(tmp_path / "perf.txt"), 2, "bfloat16", 128, 4, "graph", events, [output])
    if error:
        with pytest.raises(RuntimeError, match=error):
            load("_record_vllm_evidence")(*arguments)
    else:
        load("_record_vllm_evidence")(*arguments)
    evidence = json.loads((tmp_path / "perf.txt.rank2.evidence.jsonl").read_text())
    assert evidence["message_size"] == 128
    assert evidence["numerical_passed"] == (value == 4)


def test_comm_stops_after_first_failed_child(tmp_path):
    for name, body in {
        "nvidia-smi": "printf 'B200\\nB200\\n'",
        "python3": "exit 23",
        "torchrun": 'touch "$MARKER"',
        "all_reduce_perf": "exit 0",
        "all_gather_perf": "exit 0",
        "alltoall_perf": "exit 0",
        "reduce_scatter_perf": "exit 0",
    }.items():
        script = tmp_path / name
        script.write_text("#!/bin/sh\n" + body + "\n")
        script.chmod(0o755)
    marker = tmp_path / "must-not-run"
    result = subprocess.run(
        ["bash", str(NETWORK / "collect_comm.sh"), "--all_reduce_backend", "vllm"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "MARKER": str(marker)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23
    assert not marker.exists()
    assert "All benchmarks completed" not in result.stdout
