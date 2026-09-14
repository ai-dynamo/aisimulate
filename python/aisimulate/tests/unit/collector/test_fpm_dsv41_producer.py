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
            "try:\n"
            + (" import dsv41_scheduler\n" if adapter_first else "")
            + " from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler\n"
            + " assert InstrumentedScheduler.__name__ == 'DeepseekV41RealKVScheduler'\n"
            + "finally:\n print('normal process cleanup ran', flush=True)\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if valid_source:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 1
        assert "Traceback (most recent call last)" in result.stderr
        assert "normal process cleanup ran" in result.stdout
        assert "pinned source mismatch" in result.stderr


@pytest.mark.parametrize("failure", ["source", "sdk", "activation"])
def test_preflight_subprocess_retains_failure_audit_traceback_and_cleanup(tmp_path, failure):
    runtime = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
    shutil.copy2(runtime / "preflight.py", tmp_path / "preflight.py")
    if failure != "activation":
        shutil.copy2(runtime / "dsv41/sitecustomize.py", tmp_path / "sitecustomize.py")
    native = tmp_path / "dynamo/vllm/instrumented_scheduler.py"
    native.parent.mkdir(parents=True)
    native.write_text("class BenchmarkPoint: pass\nclass InstrumentedScheduler: pass\n")
    (tmp_path / "dsv41_scheduler.py").write_text(
        "import dynamo.vllm.instrumented_scheduler as native\n"
        "class DeepseekV41RealKVScheduler(native.InstrumentedScheduler): pass\n"
        "native.InstrumentedScheduler = DeepseekV41RealKVScheduler\n"
    )
    (tmp_path / "runtime-source-sha256.json").write_text(
        json.dumps(
            {
                "dynamo/vllm/instrumented_scheduler.py": "0" * 64
                if failure == "source"
                else hashlib.sha256(native.read_bytes()).hexdigest(),
            }
        )
    )
    # A regular private package shadows the installed SDK so this test proves
    # the exact preflight error path without any model/backend import.
    (tmp_path / "aiconfigurator_core").mkdir()
    (tmp_path / "aiconfigurator_core/__init__.py").write_text("raise ImportError('SDK native extension unavailable')\n")
    audit = tmp_path / "audit.json"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import preflight\nfrom pathlib import Path\n"
            f"preflight._AUDIT_PATH = Path({str(audit)!r})\n"
            "try:\n preflight.main()\nfinally:\n print('preflight cleanup ran', flush=True)\n",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path), DYN_FPM_DSV41_REAL_KV="1"),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 1
    assert "Traceback (most recent call last)" in child.stderr
    assert "preflight cleanup ran" in child.stdout
    receipt = json.loads(audit.read_text())
    assert receipt["status"] == "failed"
    expected = {
        "source": "pinned source mismatch",
        "sdk": "SDK native extension unavailable",
        "activation": "source-checked scheduler activation did not occur",
    }[failure]
    assert expected in receipt["import_error"] and expected in child.stderr
