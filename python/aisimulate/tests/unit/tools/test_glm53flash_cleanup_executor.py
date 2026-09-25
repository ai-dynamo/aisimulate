# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY execute pinned cleanup Python control flow with fake Slurm commands."""

import copy
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
        stdout = f"{f.job}.0|{f.step}\n999.0|{f.step}\n" if len(calls) == 2 else ""
        if argv[0] == "scontrol":
            stdout = "ClusterName = " + f.cluster + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)), patch.dict(os.environ, {"SLURM_JOB_ID": "999"}):
        proof = f.run(tmp_path / "output", cleanup=executor.cleanup_original)
        assert os.environ["SLURM_JOB_ID"] == "999"
    assert calls == [
        ["scontrol", "--local", "show", "config"],
        c.queue_argv(f.cluster),
        c.cancel_argv(f.cluster, f.job + ".0"),
        c.queue_argv(f.cluster),
    ]
    assert proof["cleanup"]["commands"][0]["stdout"].endswith("999.0|" + f.step + "\n")


@pytest.mark.parametrize("bad", ["timeout", "malformed_query", "unexpired_step"])
def test_pinned_failure_never_reads_original_raw_and_keeps_outside_diagnostics(tmp_path, bad):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 0, "ClusterName = " + f.cluster + "\n", "")
        if bad == "timeout" or len(calls) > 3:
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


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "ClusterName = other\n",
        "ClusterName = all\n",
        "ClusterName = TEST_ONLY_original_cluster\nClusterName = other\n",
    ],
)
def test_wrong_unknown_or_multiple_cluster_stops_before_step_query_and_raw(tmp_path, stdout):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with (
        patch.dict(sys.modules, pinned_modules(command)),
        patch.dict(os.environ, {"SLURM_CLUSTER_NAME": f.cluster}),
        pytest.raises(ValueError, match="cluster"),
    ):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert calls == [["scontrol", "--local", "show", "config"]]
    assert f.events == []
    assert (tmp_path / "output/cluster-command.json").exists()
    assert not (tmp_path / "output/receipt.json").exists()


@pytest.mark.parametrize(
    "name", ["SLURM_CLUSTERS", "SCONTROL_FEDERATION", "SQUEUE_FEDERATION", "SQUEUE_NAMES", "SCANCEL_SIBLING"]
)
def test_inherited_target_or_filter_cannot_hide_original_steps(tmp_path, name):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        raise AssertionError("TEST_ONLY scheduler must not be contacted")

    with (
        patch.dict(sys.modules, pinned_modules(command)),
        patch.dict(os.environ, {name: "TEST_ONLY"}),
        pytest.raises(ValueError, match="inherited Slurm"),
    ):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert calls == f.events == []


def test_cluster_query_timeout_is_preserved_and_is_not_absence(tmp_path):
    f = Fixture(tmp_path)

    def command(argv, **kwargs):
        assert argv == ["scontrol", "--local", "show", "config"]
        raise subprocess.TimeoutExpired(argv, 60, output="TEST_ONLY unavailable")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(subprocess.TimeoutExpired):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert f.events == []
    assert (tmp_path / "output/cluster-command-failure.json").exists()


def test_unknown_cluster_banner_does_not_establish_absence(tmp_path):
    f = Fixture(tmp_path)

    def command(argv, **kwargs):
        stdout = "ClusterName = " + f.cluster + "\n" if argv[0] == "scontrol" else "CLUSTER: unknown\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(ValueError, match="malformed"):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert f.events == []


def local_setup(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    f.routing_mode = "verified-local"
    config = tmp_path / "TEST_ONLY_slurm.conf"
    config.write_text("TEST_ONLY immutable route")
    monkeypatch.setenv("SLURM_CONF", str(config))
    executables = {}
    for name in ("scontrol", "squeue", "scancel"):
        path = tmp_path / name
        path.write_text("TEST_ONLY executable bytes " + name)
        executables[name] = str(path)
    monkeypatch.setattr(executor.shutil, "which", lambda name: executables[name])
    fields = {
        "ClusterName": f.cluster,
        "SlurmctldAddr": "(null)",
        "SlurmctldPort": "6817",
        "SLURM_CONF": str(config),
        "FederationParameters": "(null)",
        "CommunicationParameters": "(null)",
        "AuthType": "auth/munge",
        "CliFilterPlugins": "(null)",
        "SlurmctldHost[0]": "TEST_ONLY_host(127.0.0.1)",
    }
    return f, config, fields, executables


def test_verified_local_is_explicit_and_checks_each_operation_before_and_after(tmp_path, monkeypatch):
    f, _, fields, _ = local_setup(tmp_path, monkeypatch)
    calls, operations = [], []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            stdout = "\n".join(k + " = " + v for k, v in fields.items()) + "\n"
        else:
            operations.append(argv)
            stdout = f"{f.job}.0|{f.step}\n999.0|{f.step}\n" if len(operations) == 1 else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)):
        proof = f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert operations == [
        c.queue_argv(f.cluster, "verified-local"),
        ["scancel", f.job + ".0"],
        c.queue_argv(f.cluster, "verified-local"),
    ]
    assert len([a for a in calls if a[0] == "scontrol"]) == 6
    assert len(proof["cleanup"]["local_checks"]) == 3
    assert f.events == ["inventory", "native", "inventory"]
    bad = copy.deepcopy(proof)
    bad["cleanup"]["local_checks"][-1]["after"]["state"]["environment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="configuration/environment changed"):
        c.verify(bad)
    bad = copy.deepcopy(proof)
    bad["cleanup"]["local_checks"].pop()
    with pytest.raises(ValueError, match="before/after"):
        c.verify(bad)


@pytest.mark.parametrize(
    "change",
    [
        "cluster_after_query",
        "config_after_query",
        "client_file_after_query",
        "environment_after_query",
        "executable_after_query",
        "cluster_before_cancel",
    ],
)
def test_verified_local_change_never_cancels_or_reads_raw(tmp_path, monkeypatch, change):
    f, config, fields, executables = local_setup(tmp_path, monkeypatch)
    calls = []
    configs = 0

    def command(argv, **kwargs):
        nonlocal configs
        calls.append(argv)
        if argv[0] == "scontrol":
            configs += 1
            if change == "cluster_before_cancel" and configs == 3:
                fields["ClusterName"] = "other"
            stdout = "\n".join(k + " = " + v for k, v in fields.items()) + "\n"
        else:
            assert argv[0] == "squeue", "must reject before any cancellation"
            stdout = f"{f.job}.0|{f.step}\n"
            if change == "cluster_after_query":
                fields["ClusterName"] = "other"
            if change == "config_after_query":
                fields["SlurmctldHost[0]"] = "TEST_ONLY_changed"
            if change == "client_file_after_query":
                config.write_text("TEST_ONLY changed")
            if change == "environment_after_query":
                monkeypatch.setenv("SLURM_CONF_SERVER", "TEST_ONLY_changed")
            if change == "executable_after_query":
                Path(executables["scancel"]).write_text("TEST_ONLY changed")
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(ValueError, match="cluster|changed"):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert not any(a[0] == "scancel" for a in calls)
    assert f.events == []
    assert not (tmp_path / "output/receipt.json").exists()


def test_explicit_cluster_failure_does_not_fall_back_to_local(tmp_path):
    f = Fixture(tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 0, "ClusterName = " + f.cluster + "\n", "")
        raise subprocess.CalledProcessError(1, argv, output="", stderr="TEST_ONLY unavailable")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(subprocess.CalledProcessError):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert calls == [["scontrol", "--local", "show", "config"], c.queue_argv(f.cluster)]
    assert f.events == []


def test_verified_local_post_cancel_cluster_change_invalidates_proof(tmp_path, monkeypatch):
    f, _, fields, _ = local_setup(tmp_path, monkeypatch)
    cancellations = []

    def command(argv, **kwargs):
        if argv[0] == "scontrol":
            stdout = "\n".join(k + " = " + v for k, v in fields.items()) + "\n"
        elif argv[0] == "squeue":
            stdout = f"{f.job}.0|{f.step}\n"
        else:
            assert argv == ["scancel", f.job + ".0"]
            cancellations.append(argv)
            fields["ClusterName"] = "TEST_ONLY_changed_after_cancel"
            stdout = ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(ValueError, match="cluster"):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert cancellations == [["scancel", f.job + ".0"]]
    assert f.events == []
    assert (tmp_path / "output/failure.json").is_file()
    assert not (tmp_path / "output/receipt.json").exists()


@pytest.mark.parametrize("issue", ["missing_host", "federation"])
def test_verified_local_unqualified_configuration_stops_before_queue(tmp_path, monkeypatch, issue):
    f, _, fields, _ = local_setup(tmp_path, monkeypatch)
    if issue == "missing_host":
        fields.pop("SlurmctldHost[0]")
    else:
        fields["FederationParameters"] = "fed_display"
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "\n".join(k + " = " + v for k, v in fields.items()), "")

    with patch.dict(sys.modules, pinned_modules(command)), pytest.raises(ValueError, match="configuration|federated"):
        f.run(tmp_path / "output", cleanup=executor.cleanup_original)
    assert calls == [["scontrol", "--local", "show", "config"]]
    assert f.events == []
