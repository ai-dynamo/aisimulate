# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY execute pinned cleanup Python control flow with fake Slurm commands."""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from .test_glm53flash_cleanup_reconciliation import Fixture, c, executor

pytestmark = pytest.mark.unit

SOURCE = Path(__file__).resolve().parents[3] / "collector/fpm_forward/slurm.py"


def pinned_modules(command):
    assert c.digest(SOURCE.read_bytes()) == c.SLURM_SOURCE
    root = types.ModuleType("collector")
    root.__path__ = []
    package = types.ModuleType("collector.fpm_forward")
    package.__path__ = []
    runner = types.ModuleType("collector.fpm_forward.runner")
    runner._run_command = command
    fpm_contract = types.ModuleType("aisimulate.fpm_contract")
    fpm_contract.FPM_BENCHMARK_RESULT_GLOB = "TEST_ONLY_unused"
    spec = importlib.util.spec_from_file_location("collector.fpm_forward.slurm", SOURCE)
    slurm = importlib.util.module_from_spec(spec)
    modules = {
        "collector": root,
        "collector.fpm_forward": package,
        "collector.fpm_forward.runner": runner,
        "collector.fpm_forward.slurm": slurm,
        "aisimulate.fpm_contract": fpm_contract,
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(slurm)
    return modules


def test_pinned_algorithm_only_cancels_original_observed_step_and_keeps_current_allocation_env(tmp_path):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        assert kwargs == {"timeout": 60, "check": True}
        stdout = f"{f.job}.0|{f.step}\n999.0|{f.step}\n" if len(calls) == 1 else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)), patch.dict(os.environ, {"SLURM_JOB_ID": "999"}):
        proof = f.run(tmp_path / "output", cleanup=executor.cleanup_original)
        assert os.environ["SLURM_JOB_ID"] == "999"
    assert calls == [
        ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"],
        ["scancel", f.job + ".0"],
        ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"],
    ]
    assert proof["cleanup"]["commands"][0]["stdout"].endswith("999.0|" + f.step + "\n")


@pytest.mark.parametrize("bad", ["timeout", "malformed_query", "unexpired_step"])
def test_pinned_failure_never_reads_original_raw_and_keeps_outside_diagnostics(tmp_path, bad):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if bad == "timeout" or len(calls) > 2:
            raise subprocess.TimeoutExpired(
                argv, 60, output="TEST_ONLY original stdout", stderr="TEST_ONLY unavailable"
            )
        value = "MALFORMED" if bad == "malformed_query" else f"{f.job}.0|{f.step}\n"
        return subprocess.CompletedProcess(argv, 0, value, "")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises((ValueError, subprocess.TimeoutExpired)):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert f.events == []
    assert (tmp_path / "output/failure.json").is_file()
    assert not (tmp_path / "output/receipt.json").exists()
    assert list((tmp_path / "output").glob("cleanup-command-*.json"))
    assert not (f.cell / "logs/transport-failures").exists()


def test_isolated_public_executor_help_does_not_need_collector_or_scheduler():
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(executor.__file__)), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--request" in result.stdout


def test_isolated_public_executor_rejects_invalid_request_without_output(tmp_path):
    request = tmp_path / "TEST_ONLY_bad.json"
    request.write_text("{}")
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(executor.__file__)), "--request", str(request), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "request fields differ" in result.stderr
    assert not output.exists()
