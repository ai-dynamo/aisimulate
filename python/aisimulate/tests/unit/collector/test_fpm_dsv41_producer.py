# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run producer state-machine stubs in an isolated process."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_real_kv_producer_lifecycle():
    root = Path(__file__).resolve().parents[3]
    runtime = root / "collector/fpm_forward/runtime/dsv41"
    environment = dict(os.environ)
    environment["AIC_FPM_DSV41_PRODUCER"] = str(runtime / "dsv41_scheduler.py")
    environment["AIC_FPM_NATIVE_ARTIFACT"] = str(root / "collector/fpm_forward/native_artifact.py")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fixtures") / "dsv41_producer_lifecycle.py")],
        env=environment,
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("valid_source", [False, True])
@pytest.mark.parametrize("adapter_first", [False, True])
def test_bootstrap_is_lazy_in_helpers_and_fail_closed_at_scheduler_import(tmp_path, valid_source, adapter_first):
    runtime = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime/dsv41"
    shutil.copy2(runtime / "sitecustomize.py", tmp_path / "sitecustomize.py")
    module = tmp_path / "dynamo/vllm/instrumented_scheduler.py"
    module.parent.mkdir(parents=True)
    # A scheduler can spawn compiler/helper interpreters that inherit the same
    # environment; those children must not recursively import the scheduler.
    helper = "import sys; assert 'dynamo.vllm.instrumented_scheduler' not in sys.modules"
    module.write_text(
        "import subprocess,sys\n"
        f"subprocess.run([sys.executable, '-c', {helper!r}], check=True, timeout=5)\n"
        "class InstrumentedScheduler: pass\n"
    )
    (tmp_path / "dsv41_scheduler.py").write_text(
        "import dynamo.vllm.instrumented_scheduler as native\n"
        "class DeepseekV41RealKVScheduler(native.InstrumentedScheduler): pass\n"
        "native.InstrumentedScheduler = DeepseekV41RealKVScheduler\n"
    )
    (tmp_path / "runtime-source-sha256.json").write_text(
        json.dumps(
            {
                "dynamo/vllm/instrumented_scheduler.py": hashlib.sha256(module.read_bytes()).hexdigest()
                if valid_source
                else "0" * 64
            }
        )
    )
    environment = dict(os.environ, PYTHONPATH=str(tmp_path), DYN_FPM_DSV41_REAL_KV="1")
    child = subprocess.run([sys.executable, "-c", helper], env=environment, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import dsv41_scheduler; " if adapter_first else "")
            + "from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler; "
            + "assert InstrumentedScheduler.__name__ == 'DeepseekV41RealKVScheduler'",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if valid_source:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 78
        assert "pinned source mismatch" in result.stderr
