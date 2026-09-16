# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU subprocess regressions for import isolation and process-group ownership."""

import itertools
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from collector.campaigns import run_shard as runner

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="Slurm runner uses POSIX process groups")]


def wait_for_file(path, proc, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert proc.poll() is None, f"Process exited before creating {path}"
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def emergency_cleanup(pid):
    if pid is not None:
        runner.signal_process_group(pid, signal.SIGKILL)


def test_parent_sitecustomize_and_extra_modules_are_not_imported(tmp_path, monkeypatch):
    source = tmp_path / "source/python/aisimulate"
    source.mkdir(parents=True)
    (source / "attested_test_module.py").write_text("VALUE = 7\n")
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (untrusted / "sitecustomize.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    (untrusted / "unattested_aisim_test_module.py").write_text("VALUE = 99\n")
    monkeypatch.setenv("PYTHONPATH", str(untrusted))
    monkeypatch.setenv("PYTHONHOME", str(untrusted))
    monkeypatch.setenv("PYTHONUSERBASE", str(untrusted))
    monkeypatch.setenv("PYTHONSTARTUP", str(untrusted / "sitecustomize.py"))
    env = runner.collector_environment(tmp_path, {"path": "/attested/runtime", "sha256": "a" * 64})
    assert env["PYTHONPATH"] == str(source)
    assert all(key not in env for key in ("PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, site, attested_test_module; "
            "assert attested_test_module.VALUE == 7; "
            "assert importlib.util.find_spec('unattested_aisim_test_module') is None; "
            "assert site.ENABLE_USER_SITE is False",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=5,
    )
    assert not marker.exists()


@pytest.mark.parametrize("ignores_term", [False, True])
def test_exception_stops_and_reaps_real_collector(tmp_path, ignores_term):
    marker = tmp_path / "ready"
    child = "import pathlib,signal,time; "
    if ignores_term:
        child += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    child += f"pathlib.Path({str(marker)!r}).write_text('ready'); time.sleep(60)"
    status = {}
    failure = RuntimeError("checkpoint inspection failed")
    proc = None
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        with (
            pytest.raises(RuntimeError) as caught,
            runner.collector_process(
                [sys.executable, "-c", child],
                data=tmp_path,
                env=os.environ.copy(),
                log=subprocess.DEVNULL,
                status=status,
                shutdown_timeout=0.15,
            ) as proc,
        ):
            wait_for_file(marker, proc)
            raise failure
        assert caught.value is failure
        assert proc.returncode in (-signal.SIGTERM, -signal.SIGKILL)
        assert status["process_group_cleanup"]["forced"] is ignores_term
        assert signal.getsignal(signal.SIGTERM) == previous_handler
    finally:
        if proc is not None:
            emergency_cleanup(proc.pid)
            proc.wait(timeout=5)


def test_exited_leader_does_not_leave_stubborn_worker(tmp_path):
    marker = tmp_path / "worker-ready"
    worker = (
        "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(marker)!r}).write_text('ready'); time.sleep(60)"
    )
    leader = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{worker!r}]); "
        f"p=pathlib.Path({str(marker)!r}); "
        "\nwhile not p.exists(): time.sleep(.01)\n"
    )
    status = {}
    proc = None
    try:
        with runner.collector_process(
            [sys.executable, "-c", leader],
            data=tmp_path,
            env=os.environ.copy(),
            log=subprocess.PIPE,
            status=status,
            shutdown_timeout=0.15,
        ) as proc:
            assert proc.wait(timeout=5) == 0
        assert status["process_group_cleanup"]["forced"] is True
        assert not runner.process_group_active(proc.pid)
        # The worker inherited the pipe. EOF proves it no longer holds the
        # descriptor even though its process-group leader exited first.
        proc.communicate(timeout=5)
    finally:
        if proc is not None:
            emergency_cleanup(proc.pid)
            proc.wait(timeout=5)
            if proc.stdout is not None:
                proc.stdout.close()


@pytest.mark.parametrize("failure_point", ["checkpoint", "proc_inspection", "status_write"])
def test_monitoring_failures_are_recorded_after_cleanup(tmp_path, monkeypatch, failure_point):
    marker = tmp_path / "ready"
    child = f"import pathlib,time; pathlib.Path({str(marker)!r}).write_text('ready'); time.sleep(60)"
    status_path = tmp_path / "status.json"
    status = {"started_unix": time.time()}
    failure = OSError("injected " + failure_point)
    original_save = runner.save

    def fail(*args, **kwargs):
        raise failure

    if failure_point == "checkpoint":
        monkeypatch.setattr(runner, "checkpoint_state", fail)
    elif failure_point == "proc_inspection":
        clock = itertools.count(100, 100)
        monkeypatch.setattr(
            runner, "time", SimpleNamespace(time=time.time, monotonic=lambda: next(clock), sleep=time.sleep)
        )
        monkeypatch.setattr(runner, "checkpoint_state", lambda _: ({}, {"done"}, set()))
        monkeypatch.setattr(runner, "finalized_all_ids", lambda *args: True)
        monkeypatch.setattr(runner, "reap_finished_workers", fail)
    else:

        def save(path, document):
            if document["status"] == "running":
                raise failure
            return original_save(path, document)

        monkeypatch.setattr(runner, "save", save)
    try:
        with pytest.raises(OSError) as caught:
            runner.execute_collector(
                [sys.executable, "-c", child],
                tmp_path,
                status_path,
                status,
                {"planned_tasks": 1},
                False,
                os.environ.copy(),
                subprocess.DEVNULL,
                poll_interval=0.01,
                shutdown_timeout=0.15,
            )
        assert caught.value is failure
        saved = json.loads(status_path.read_text())
        assert saved["status"] == "runner_failed"
        assert saved["runner_error"]["message"] == str(failure)
        assert saved["collector_exit_code"] is not None
        assert "process_group_cleanup" in saved
    finally:
        emergency_cleanup(status.get("collector_pid"))


def test_failure_recording_cannot_mask_original_error(tmp_path, monkeypatch, capsys):
    status = {"started_unix": time.time()}
    original = OSError("initial status write failed")
    subsequent = RuntimeError("failure report write failed")

    def save(path, document):
        raise original if document["status"] == "running" else subsequent

    monkeypatch.setattr(runner, "save", save)
    try:
        with pytest.raises(OSError) as caught:
            runner.execute_collector(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                tmp_path,
                tmp_path / "status.json",
                status,
                {"planned_tasks": 1},
                False,
                os.environ.copy(),
                subprocess.DEVNULL,
                poll_interval=0.01,
                shutdown_timeout=0.15,
            )
        assert caught.value is original
        assert status["status"] == "runner_failed"
        assert status["collector_exit_code"] is not None
        assert "could not persist failure status" in capsys.readouterr().err
    finally:
        emergency_cleanup(status.get("collector_pid"))


def test_successful_collector_is_reaped_before_return(tmp_path):
    checkpoint = tmp_path / "checkpoint/vllm"
    checkpoint.mkdir(parents=True)
    checkpoint_path = str(checkpoint / "case.json")
    payload = json.dumps({"done": ["case"], "failed": []})
    script = f"from pathlib import Path; Path({checkpoint_path!r}).write_text({payload!r})"
    status = {"started_unix": time.time()}
    done, failed = runner.execute_collector(
        [sys.executable, "-c", script],
        tmp_path,
        tmp_path / "status.json",
        status,
        {"planned_tasks": 1},
        False,
        os.environ.copy(),
        subprocess.DEVNULL,
        poll_interval=0.01,
        shutdown_timeout=0.15,
    )
    assert done == {"case"} and not failed
    assert status["collector_exit_code"] == 0
    assert "process_group_cleanup" in status
    assert "runner_error" not in status


def test_controlled_walltime_yield_also_terminates_session(tmp_path):
    status = {"started_unix": time.time() - 6601}
    try:
        runner.execute_collector(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            tmp_path / "status.json",
            status,
            {"planned_tasks": 1},
            False,
            os.environ.copy(),
            subprocess.DEVNULL,
            poll_interval=0.01,
            shutdown_timeout=0.15,
        )
        assert status["stop_reason"] == "walltime_checkpoint_yield"
        assert status["collector_exit_code"] is not None
        assert "process_group_cleanup" in status
    finally:
        emergency_cleanup(status.get("collector_pid"))


def test_secondary_cleanup_error_does_not_mask_monitor_error(tmp_path, monkeypatch):
    status = {"started_unix": time.time()}
    failure = ValueError("bad checkpoint")
    original_cleanup = runner.terminate_process_group

    def fail_checkpoint(*args):
        raise failure

    def cleanup_then_fail(proc, **kwargs):
        original_cleanup(proc, **kwargs)
        raise OSError("cleanup reporting failed")

    monkeypatch.setattr(runner, "checkpoint_state", fail_checkpoint)
    monkeypatch.setattr(runner, "terminate_process_group", cleanup_then_fail)
    try:
        with pytest.raises(ValueError) as caught:
            runner.execute_collector(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                tmp_path,
                tmp_path / "status.json",
                status,
                {"planned_tasks": 1},
                False,
                os.environ.copy(),
                subprocess.DEVNULL,
                poll_interval=0.01,
                shutdown_timeout=0.15,
            )
        assert caught.value is failure
        saved = json.loads((tmp_path / "status.json").read_text())
        assert saved["runner_error"]["message"] == "bad checkpoint"
        assert "cleanup reporting failed" in saved["cleanup_error"]
        assert saved["collector_exit_code"] is not None
    finally:
        emergency_cleanup(status.get("collector_pid"))


def test_sigterm_to_runner_cleans_up_its_collector(tmp_path):
    marker = tmp_path / "collector-ready"
    status_path = tmp_path / "status.json"
    collector = (
        "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(marker)!r}).write_text('ready'); time.sleep(60)"
    )
    script = (
        "import os,pathlib,subprocess,sys,time; from collector.campaigns.run_shard import execute_collector; "
        f"execute_collector([sys.executable,'-c',{collector!r}], pathlib.Path({str(tmp_path)!r}), "
        f"pathlib.Path({str(status_path)!r}), {{'started_unix':time.time()}}, {{'planned_tasks':1}}, False, "
        "os.environ.copy(), subprocess.DEVNULL, poll_interval=.01, shutdown_timeout=.15)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        start_new_session=True,
    )
    child_pid = None
    try:
        wait_for_file(marker, proc)
        child_pid = json.loads(status_path.read_text())["collector_pid"]
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=5)
        assert proc.returncode != 0
        status = json.loads(status_path.read_text())
        assert status["status"] == "runner_failed"
        assert status["runner_error"]["type"] == "InterruptedError"
        assert status["collector_exit_code"] == -signal.SIGKILL
        assert status["process_group_cleanup"]["forced"] is True
        assert b"Runner received signal" in stderr
    finally:
        emergency_cleanup(child_pid)
        emergency_cleanup(proc.pid)
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@pytest.mark.parametrize("states,active", [("123 Z\n", False), ("123 S\n", True), ("456 R\n", False)])
def test_group_inspection_distinguishes_zombies_from_live_writers(monkeypatch, states, active):
    monkeypatch.setattr(runner.os, "killpg", lambda *args: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: states)
    assert runner.process_group_active(123) is active


@pytest.mark.parametrize("active", [False, True])
def test_signal_permission_error_requires_confirming_group_exit(monkeypatch, active):
    def denied(*args):
        raise PermissionError("cannot signal group")

    monkeypatch.setattr(runner.os, "killpg", denied)
    monkeypatch.setattr(runner, "process_group_active", lambda pid: active)
    if active:
        with pytest.raises(PermissionError, match="cannot signal group"):
            runner.signal_process_group(123, signal.SIGTERM)
    else:
        runner.signal_process_group(123, signal.SIGTERM)


def test_live_group_after_sigkill_fails_closed(monkeypatch):
    clock = itertools.count(100, 100)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None))
    monkeypatch.setattr(runner, "process_group_active", lambda pid: True)
    signals = []
    monkeypatch.setattr(runner, "signal_process_group", lambda pid, signum: signals.append(signum))
    proc = SimpleNamespace(pid=123, returncode=0, wait=lambda **kwargs: 0)
    with pytest.raises(TimeoutError, match="remains active after SIGKILL"):
        runner.terminate_process_group(proc, timeout=0.15)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_startup_failure_is_reported_without_masking_original(tmp_path):
    status = {"started_unix": time.time()}
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(FileNotFoundError):
        runner.execute_collector(
            [str(tmp_path / "missing-collector")],
            tmp_path,
            tmp_path / "status.json",
            status,
            {"planned_tasks": 1},
            False,
            os.environ.copy(),
            subprocess.DEVNULL,
        )
    assert status["status"] == "runner_failed"
    assert status["runner_error"]["type"] == "FileNotFoundError"
    assert signal.getsignal(signal.SIGTERM) == previous_handler


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_during_startup_does_not_lose_child_handle(tmp_path, monkeypatch, signum):
    popen = subprocess.Popen
    children = []
    previous_handler = signal.getsignal(signum)

    def start_then_signal(*args, **kwargs):
        proc = popen(*args, **kwargs)
        if kwargs.get("start_new_session"):
            children.append(proc)
            os.kill(os.getpid(), signum)
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", start_then_signal)
    status = {}
    try:
        with (
            pytest.raises(InterruptedError, match="Runner received signal"),
            runner.collector_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                data=tmp_path,
                env=os.environ.copy(),
                log=subprocess.DEVNULL,
                status=status,
                shutdown_timeout=0.15,
            ),
        ):
            pytest.fail("Startup interruption must not start monitoring")
        assert len(children) == 1
        assert children[0].returncode is not None
        assert "process_group_cleanup" in status
        assert signal.getsignal(signum) == previous_handler
    finally:
        monkeypatch.setattr(runner.subprocess, "Popen", popen)
        for proc in children:
            emergency_cleanup(proc.pid)
            proc.wait(timeout=5)
