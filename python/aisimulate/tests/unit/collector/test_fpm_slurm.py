# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from collector.fpm_forward.slurm import SlurmCellRunner

pytestmark = pytest.mark.unit


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "test-node")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        json.dumps(
            {
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "cell"},
                "spec": {"replicas": 1, "leaderWorkerTemplate": {"size": 1}},
            }
        )
    )
    return SlurmCellRunner(manifest, tmp_path, image="image@sha256:abc", mounts=("/cache:/cache",), total_gpus=4)


def test_slurm_requires_scheduler_allocation(runner, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="existing sbatch/salloc"):
        SlurmCellRunner(runner.cell_dir / "manifest.yaml", runner.cell_dir, image=runner.image, mounts=(), total_gpus=4)


def test_slurm_stage_and_argv_keep_shared_result_unit_identity(runner, monkeypatch):
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(
            stdout="JobId=1234 JobState=RUNNING NodeList=test-node" if "job" in args else "test-node\n", stderr=""
        )

    monkeypatch.setattr(runner, "_command", command)
    units = runner.wait_ready(1)
    source = runner.cell_dir / "fpm_exec.sh"
    source.write_text("exit 0\n")
    startup = runner.cell_dir / "collector-runtime-env.sh"
    startup.write_text("export FPM_READINESS_TIMEOUT_SECONDS=600\n")
    runner.stage(units, [source, startup])
    assert (runner.cell_dir / "slurm-runtime" / startup.name).read_bytes() == startup.read_bytes()
    runner._exec(units[0], ["bash", "/tmp/fpm-bench/fpm_exec.sh"], timeout=10)
    assert units == ["node0000"]
    assert (runner.cell_dir / "slurm-runtime" / source.name).read_text() == source.read_text()
    argv = commands[-1]
    assert "--jobid=1234" in argv and "--gpus-per-node=4" in argv
    assert "FPM_NODE_RANK=0" in argv and "FPM_MASTER_ADDR=test-node" in argv
    assert f"{runner.cell_dir}/raw/node0000:/results" in next(a for a in argv if a.startswith("--container-mounts="))


@pytest.mark.skipif(os.name != "posix", reason="native session creation is POSIX-specific")
@pytest.mark.parametrize("exit_code", [0, 17])
def test_slurm_native_exec_chain_can_create_session_and_preserves_argv(runner, monkeypatch, exit_code):
    """Reproduce Pyxis's process-group leader without requiring Slurm or GPUs."""
    literal = 'spaces; $(not-a-command) "quoted"'
    probe = (
        "import json,os,sys; before=[os.getpid(),os.getpgrp(),os.getsid(0)]; "
        "os.setsid(); print(json.dumps({'before':before,'after':[os.getpid(),os.getpgrp(),os.getsid(0)],"
        "'argv':sys.argv[1:],'rank':os.environ['FPM_NODE_RANK']})); sys.exit(int(sys.argv[1]))"
    )

    def run_actual_container_command(args, **kwargs):
        return subprocess.run(
            args[args.index("env") :],
            start_new_session=True,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    runner.hosts = ["test-node"]
    monkeypatch.setattr(runner, "_command", run_actual_container_command)
    result = runner._exec(
        "node0000",
        ["bash", "-c", 'exec "$@"', "native-launcher", sys.executable, "-c", probe, str(exit_code), literal],
        timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["before"][0] != receipt["before"][1]
    assert len(set(receipt["after"])) == 1
    assert receipt["argv"] == [str(exit_code), literal]
    assert receipt["rank"] == "0"


def test_slurm_cleanup_cancels_only_receipted_steps_and_verifies_exit(runner, monkeypatch):
    runner.owner_path.parent.mkdir(parents=True, exist_ok=True)
    runner.owner_path.write_text(json.dumps({"job_id": "1233", "step_name": runner.step_name}))
    commands = []
    snapshots = iter(
        [f"1234.2|{runner.step_name}\n1233.1|{runner.step_name}\n1234.4|other\n1234.batch|{runner.step_name}\n", ""]
    )

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout=next(snapshots) if args[0] == "squeue" else "", stderr="")

    monkeypatch.setattr(runner, "_command", command)
    runner.cleanup()
    assert [args for args in commands if args[0] == "scancel"] == [["scancel", "1234.2"], ["scancel", "1233.1"]]
    assert len([args for args in commands if args[0] == "squeue"]) == 2


def test_slurm_cleanup_reports_leaked_steps(runner, monkeypatch):
    monkeypatch.setattr(
        runner, "_command", lambda *a, **k: SimpleNamespace(stdout=f"1234.2|{runner.step_name}", stderr="")
    )
    clock = iter([0, 61])
    monkeypatch.setattr("collector.fpm_forward.slurm.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="remain after cleanup"):
        runner.cleanup()


def test_slurm_refuses_allocation_geometry_mismatch(runner, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_command",
        lambda args, **k: SimpleNamespace(
            stdout="JobId=1234 JobState=RUNNING NodeList=node-a,node-b" if "job" in args else "node-a node-b"
        ),
    )
    with pytest.raises(ValueError, match="exactly 1 allocated nodes"):
        runner.wait_ready(1)


def test_preparation_preserves_slurm_failure_streams_across_retries(runner, monkeypatch):
    runner.hosts = ["test-node"]
    errors = [
        subprocess.CalledProcessError(1, ["srun"], output="preparation started", stderr="task resource conflict"),
        subprocess.TimeoutExpired(["srun"], 300, output=b"waiting", stderr=b"container startup stalled"),
    ]
    for failure in errors:

        def fail(*args, **kwargs):
            raise failure

        monkeypatch.setattr("collector.fpm_forward.runner._run_command", fail)
        with pytest.raises(type(failure)) as caught:
            runner.prepare_attempt(runner.pods(), cell_id="cell", plan_sha256="plan", attempt_id="attempt")
        assert caught.value is failure
    records = list((runner.cell_dir / "logs" / "transport-failures").iterdir())
    assert len(records) == 2
    assert {path.joinpath("stderr.log").read_text() for path in records} == {
        "task resource conflict",
        "container startup stalled",
    }
    assert {path.joinpath("stdout.log").read_text() for path in records} == {"preparation started", "waiting"}
    assert all(json.loads(path.joinpath("failure.json").read_text())["executable"] == "srun" for path in records)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_preparation_measures_metadata_after_frozen_environment(runner, monkeypatch, backend):
    """TEST_ONLY distributions exercise Python's real metadata path selection."""
    from collector.fpm_forward import native_artifact
    from collector.fpm_forward import runner as campaign

    startup = runner.cell_dir / "frozen env ' $(not-a-command)"
    startup.mkdir()
    versions = {"image": "0.1.0", "private": "0.1.0+testonly"}
    for kind, version in versions.items():
        root = runner.cell_dir / kind
        metadata = root / f"{backend}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {backend}\nVersion: {version}\n")
    monkeypatch.setenv("PYTHONPATH", str(runner.cell_dir / "image"))
    metadata_probe = f"import importlib.metadata; print(importlib.metadata.version({backend!r}))"
    assert subprocess.check_output([sys.executable, "-c", metadata_probe], text=True).strip() == versions["image"]
    (startup / campaign.RUNTIME_ENV_FILENAME).write_text(
        f"export PYTHONPATH={shlex.quote(str(runner.cell_dir / 'private'))}\n"
    )
    destination = runner.cell_dir / "provenance.json"
    monkeypatch.setattr(campaign, "REMOTE_WORKDIR", str(startup))
    monkeypatch.setattr(native_artifact, "COLLECTOR_PROVENANCE_FILENAME", str(destination))
    runner.backend = backend
    runner.hosts = ["test-node"]
    commands = []

    def run_actual_container_command(args, **kwargs):
        commands.append(args)
        return subprocess.run(args[args.index("env") :], capture_output=True, text=True, check=True, timeout=10)

    monkeypatch.setattr(runner, "_command", run_actual_container_command)
    literal = 'spaces; $(not-a-command) "quoted"'
    runner.prepare_attempt(runner.pods(), cell_id=literal, plan_sha256=literal, attempt_id=literal)
    receipt = json.loads(destination.read_text())
    assert receipt["runtime"] == {"backend": backend, "backend_version": versions["private"]}
    assert all(receipt[key] == literal for key in ("cell_id", "plan_sha256", "attempt_id"))
    assert commands[0][commands[0].index("fpm-slurm-prepare") + 1] == str(startup / campaign.RUNTIME_ENV_FILENAME)
    assert os.environ["PYTHONPATH"] == str(runner.cell_dir / "image")


@pytest.mark.parametrize("startup_text", [None, "return 19\n"])
def test_preparation_rejects_missing_or_failed_frozen_environment(runner, monkeypatch, startup_text):
    from collector.fpm_forward import native_artifact
    from collector.fpm_forward import runner as campaign

    startup = runner.cell_dir / campaign.RUNTIME_ENV_FILENAME
    if startup_text is not None:
        startup.write_text(startup_text)
    destination = runner.cell_dir / "provenance.json"
    monkeypatch.setattr(campaign, "REMOTE_WORKDIR", str(runner.cell_dir))
    monkeypatch.setattr(native_artifact, "COLLECTOR_PROVENANCE_FILENAME", str(destination))
    runner.hosts = ["test-node"]

    def run_actual_container_command(args, **kwargs):
        return subprocess.run(args[args.index("env") :], capture_output=True, text=True, check=True, timeout=10)

    monkeypatch.setattr(runner, "_command", run_actual_container_command)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        runner.prepare_attempt(runner.pods(), cell_id="cell", plan_sha256="plan", attempt_id="attempt")
    assert caught.value.returncode == (1 if startup_text is None else 19)
    assert not destination.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_wait_ready_rejects_unbounded_or_invalid_timeout(runner, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        runner.wait_ready(1, timeout_seconds=timeout)


def test_wait_ready_waits_for_allocation_and_shares_one_deadline(runner, monkeypatch):
    clock = [0.0]
    commands = []
    states = iter(["PENDING", "CONFIGURING", "RUNNING"])
    monkeypatch.setattr("collector.fpm_forward.slurm.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "collector.fpm_forward.slurm.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def command(args, *, timeout):
        commands.append((args, timeout))
        clock[0] += 0.25
        text = f"JobId=1234 JobState={next(states)} NodeList=test-node" if "job" in args else "test-node"
        return SimpleNamespace(stdout=text)

    monkeypatch.setattr(runner, "_command", command)
    assert runner.wait_ready(1, timeout_seconds=5) == ["node0000"]
    assert [timeout for _, timeout in commands] == [5, 3.75, 2.5, 2.25]
    assert commands[-1][0] == ["scontrol", "show", "hostnames", "test-node"]


def test_wait_ready_pending_allocation_expires_without_adopting_nodes(runner, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("collector.fpm_forward.slurm.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "collector.fpm_forward.slurm.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        runner, "_command", lambda *args, **kwargs: SimpleNamespace(stdout="JobId=1234 JobState=PENDING")
    )
    with pytest.raises(TimeoutError, match="not ready before deadline: PENDING"):
        runner.wait_ready(1, timeout_seconds=0.5)
    assert clock[0] == 0.5 and runner.hosts == []


@pytest.mark.parametrize("snapshot", ["JobId=1234 JobState=FAILED", "JobId=4567 JobState=RUNNING NodeList=test-node"])
def test_wait_ready_rejects_terminal_or_foreign_allocation(runner, monkeypatch, snapshot):
    monkeypatch.setattr(runner, "_command", lambda *args, **kwargs: SimpleNamespace(stdout=snapshot))
    with pytest.raises((ValueError, RuntimeError)):
        runner.wait_ready(1)
    assert runner.hosts == []


@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("child_mode", ["cooperative", "ignore_term", "pipe_descendant"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX transport process groups")
def test_execute_interrupt_stops_live_children_before_join(runner, monkeypatch, interrupt, child_mode):
    from collector.fpm_forward import runner as campaign

    stopped = threading.Event()
    children = []
    signalled_at = []
    readiness = runner.cell_dir / "child-ready"
    monkeypatch.setattr(campaign, "_COMMAND_TERMINATION_GRACE_SECONDS", 0.2)

    def live_command(_unit, _command, *, timeout):
        child = (
            "import os,pathlib,signal,sys,time; "
            + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if child_mode != "cooperative" else "")
            + "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        )
        if child_mode == "pipe_descendant":
            # The direct transport exits on TERM; its descendant ignores TERM
            # and retains stdout/stderr, which would block communicate/join.
            child = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); time.sleep(30)"
            )
        return campaign._run_command([sys.executable, "-c", child, str(readiness)], timeout=timeout)

    def force_cleanup():
        # Keep a broken implementation bounded without trusting its helper.
        for process in children:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.kill()
        if readiness.exists():
            try:
                os.kill(int(readiness.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def interrupt_after_spawn():
        deadline = time.monotonic() + 5
        while not stopped.wait(0.01):
            with campaign._ACTIVE_COMMANDS_LOCK:
                children[:] = campaign._ACTIVE_COMMANDS
            if children and readiness.exists() and readiness.read_text():
                signalled_at.append(time.monotonic())
                os.kill(os.getpid(), interrupt)
                # Bound the failing regression as well: the broken executor
                # otherwise waits for the child's entire 30-second lifetime.
                if not stopped.wait(5):
                    force_cleanup()
                return
            if time.monotonic() >= deadline:
                return

    monkeypatch.setattr(runner, "_exec", live_command)
    interrupter = threading.Thread(target=interrupt_after_spawn)
    try:
        with pytest.raises(KeyboardInterrupt), campaign._sigterm_as_interrupt():
            interrupter.start()
            runner.execute(["node0000"])
    finally:
        returned_at = time.monotonic()
        stopped.set()
        interrupter.join(timeout=6)
        campaign.terminate_active_commands()
        force_cleanup()

    assert not interrupter.is_alive()
    assert len(children) == 1 and len(signalled_at) == 1
    assert returned_at - signalled_at[0] < 4
    expected_signal = signal.SIGKILL if child_mode == "ignore_term" else signal.SIGTERM
    assert children[0].returncode == -expected_signal
    with campaign._ACTIVE_COMMANDS_LOCK:
        assert not campaign._ACTIVE_COMMANDS


@pytest.mark.parametrize("backend", ["slurm", "kubernetes"])
@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("window", ["prelaunch", "inflight", "registered"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX transport process groups")
def test_execute_closes_concurrent_launch_admission(runner, monkeypatch, backend, interrupt, window):
    from collector.fpm_forward import runner as campaign

    arrived = threading.Barrier(3)
    released = threading.Event()
    finished = threading.Event()
    children = []
    scopes = []
    original_popen = subprocess.Popen
    original_init = campaign.CommandScope.__init__
    original_cancel = campaign.CommandScope.cancel
    monkeypatch.setattr(campaign, "_COMMAND_TERMINATION_GRACE_SECONDS", 0.1)

    def scope_init(self):
        original_init(self)
        scopes.append(self)

    def cancel(self):
        try:
            return original_cancel(self)
        finally:
            released.set()

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        if window == "inflight":
            arrived.wait(timeout=5)
            assert released.wait(5)
        return child

    def command(unit, *args, **kwargs):
        if window == "prelaunch":
            arrived.wait(timeout=5)
            assert released.wait(5)
        result = campaign._run_command([sys.executable, "-c", "import time; time.sleep(30)"], check=False)
        return (unit, result) if backend == "kubernetes" else result

    def interrupt_workers():
        try:
            if window == "registered":
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if scopes:
                        with scopes[0].lock:
                            if len(scopes[0].processes) == 2:
                                break
                    time.sleep(0.005)
                else:
                    return
            else:
                arrived.wait(timeout=5)
            os.kill(os.getpid(), interrupt)
            if not finished.wait(5):
                released.set()
                for child in children:
                    child.kill()
        except threading.BrokenBarrierError:
            released.set()

    monkeypatch.setattr(campaign.CommandScope, "__init__", scope_init)
    monkeypatch.setattr(campaign.CommandScope, "cancel", cancel)
    monkeypatch.setattr(subprocess, "Popen", popen)
    if backend == "kubernetes":
        cell_dir = runner.cell_dir
        runner = object.__new__(campaign.KubernetesCellRunner)
        runner.cell_dir = cell_dir
        monkeypatch.setattr(runner, "_run_pod", command)
    else:
        monkeypatch.setattr(runner, "_exec", command)
    interrupter = threading.Thread(target=interrupt_workers)
    try:
        with pytest.raises(KeyboardInterrupt), campaign._sigterm_as_interrupt():
            interrupter.start()
            runner.execute(["node0000", "node0001"])
    finally:
        finished.set()
        released.set()
        interrupter.join(timeout=6)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
    assert not interrupter.is_alive()
    assert len(children) == (0 if window == "prelaunch" else 2)
    assert all(child.returncode != 0 for child in children)
    assert scopes[0].cancelled and scopes[0].inflight == 0 and not scopes[0].processes
    with campaign._ACTIVE_COMMANDS_LOCK:
        assert not campaign._ACTIVE_COMMANDS
    # Cancellation cannot poison salvage, scancel, or a subsequent invocation.
    monkeypatch.setattr(subprocess, "Popen", original_popen)
    assert campaign._run_command([sys.executable, "-c", "pass"]).returncode == 0
    assert campaign.CommandScope().run(campaign._run_command, [sys.executable, "-c", "pass"]).returncode == 0


def test_cleanup_permission_error_is_reported_without_losing_interrupt(monkeypatch):
    from collector.fpm_forward import runner as campaign

    stopped = []

    class Child:
        def wait(self, timeout):
            stopped.append("wait")

    child = Child()
    monkeypatch.setattr(campaign, "_signal_command", lambda process, force: stopped.append(force))

    def denied(process):
        raise PermissionError("injected group probe denied")

    monkeypatch.setattr(campaign, "_command_group_running", denied)
    scope = campaign.CommandScope()
    scope.processes.add(child)
    error = KeyboardInterrupt()
    campaign._cancel_preserving_interrupt(scope, error)
    assert stopped == [False, True, "wait"]
    assert scope.cancelled
    assert any("PermissionError" in note for note in error.__notes__)


@pytest.mark.skipif(
    not (hasattr(os, "waitid") and hasattr(os, "WNOWAIT")),
    reason="Requires waitid without reaping to stage the real exit race",
)
def test_darwin_probe_reaps_child_that_exits_between_poll_and_killpg(monkeypatch):
    from collector.fpm_forward import runner as campaign

    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    poll, killpg = child.poll, os.killpg
    probes = []
    polls = []

    def exit_after_live_poll():
        status = poll()
        polls.append(status)
        if len(polls) == 1:
            assert status is None  # The actual process is alive at the first poll.
            child.stdin.write(b"x")
            child.stdin.flush()
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            # It is now an unreaped zombie, before the group probe runs.
        return status

    def darwin_probe(pid, sig):
        assert pid == child.pid and sig == 0
        probes.append(child.returncode)
        if child.returncode is None:
            raise PermissionError("simulated Darwin zombie-only group")
        return killpg(pid, sig)  # Real ESRCH after the direct child was reaped.

    monkeypatch.setattr(campaign, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(child, "poll", exit_after_live_poll)
    monkeypatch.setattr(os, "killpg", darwin_probe)
    try:
        assert campaign._command_group_running(child) is False
        assert polls == [None, 0]
        assert probes == [None, 0]
    finally:
        child.stdin.close()
        child.wait(timeout=3)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.skipif(not (hasattr(os, "waitid") and hasattr(os, "WNOWAIT")), reason="Requires a real unreaped child")
def test_darwin_signal_accepts_only_reaped_disappeared_group(monkeypatch, force):
    from collector.fpm_forward import runner as campaign

    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    killpg = os.killpg
    signals = []

    def darwin_signal(pid, sig):
        signals.append(sig)
        if child.returncode is None:
            raise PermissionError("simulated Darwin zombie-only group")
        return killpg(pid, sig)

    monkeypatch.setattr(campaign, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(os, "killpg", darwin_signal)
    try:
        campaign._signal_command(child, force=force)
        assert signals == [signal.SIGKILL if force else signal.SIGTERM, 0]
        assert child.returncode == 0
    finally:
        child.wait(timeout=3)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("reprobe", ["denied", "surviving_group"])
@pytest.mark.parametrize("operation", ["probe", "term", "kill"])
def test_group_permission_denial_is_not_hidden_by_direct_child_exit(monkeypatch, platform, reprobe, operation):
    from collector.fpm_forward import runner as campaign

    calls = []
    child = SimpleNamespace(pid=123456789, poll=lambda: 0)

    def denied(pid, sig):
        calls.append(sig)
        if len(calls) == 1 or reprobe == "denied":
            raise PermissionError("permission genuinely denied")
        # Direct child is gone, but another process still occupies the group.
        return None

    monkeypatch.setattr(campaign, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(os, "killpg", denied)
    with pytest.raises(PermissionError, match="genuinely denied"):
        if operation == "probe":
            campaign._command_group_running(child)
        else:
            campaign._signal_command(child, force=operation == "kill")
    assert len(calls) == (2 if platform == "darwin" else 1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX transport process groups")
def test_darwin_cleanup_waits_for_communicate_owner_to_reap(monkeypatch):
    """Portable reproduction of Darwin EPERM while communicate owns waitpid."""
    from collector.fpm_forward import runner as campaign

    owner_waiting = threading.Event()
    release_owner = threading.Event()
    children, results, failures = [], [], []
    real_popen, real_killpg = subprocess.Popen, os.killpg
    denied = []

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        real_wait = child._wait

        def wait(timeout):
            # communicate() calls this wait. Keep its owner's waitpid lock
            # across the first cleanup signal/probe without using waitid.
            with child._waitpid_lock:
                owner_waiting.set()
                assert release_owner.wait(5)
            return real_wait(timeout)

        child._wait = wait
        return child

    def killpg(pid, sig):
        if not release_owner.is_set():
            if sig == signal.SIGTERM:
                real_killpg(pid, sig)
            elif sig == 0:
                release_owner.set()
            denied.append(sig)
            raise PermissionError("Darwin zombie awaiting communicate owner")
        return real_killpg(pid, sig)

    def run():
        try:
            results.append(
                campaign._run_command(
                    [sys.executable, "-c", "import os, time; os.close(1); os.close(2); time.sleep(30)"], check=False
                )
            )
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(campaign, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(campaign, "_COMMAND_TERMINATION_GRACE_SECONDS", 1)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert owner_waiting.wait(5)
        campaign._stop_commands(children)
    finally:
        release_owner.set()
        for child in children:
            if child.poll() is None:
                real_killpg(child.pid, signal.SIGKILL)
        worker.join(5)
    assert not worker.is_alive()
    assert not failures
    assert results[0].returncode == -signal.SIGTERM
    assert denied[:2] == [signal.SIGTERM, 0]
