# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI identity, version, and command parity across canonical and legacy launchers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisim
from aisim import package_identity, supervision
from aisim.cli_args import build_parser

pytestmark = pytest.mark.unit


def _parsers(parser, path=()):
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from _parsers(child, (*path, name))


COMMAND_PATHS = tuple(path for path, _ in _parsers(build_parser()))


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/venv/bin/aisim", "aisim"),
        ("/venv/bin/aisimulate", "aisimulate"),
        ("/venv/Scripts/aisim.exe", "aisim"),
        (r"C:\venv\Scripts\aisimulate.exe", "aisimulate"),
        ("/venv/bin/aisim-script.py", "aisim"),
        ("/site-packages/aisim/__main__.py", "aisim"),
        ("/site-packages/aisimulate/__main__.py", "aisimulate"),
        ("", "aisim"),
    ],
)
def test_program_name_follows_the_invoked_command(monkeypatch, argv0, expected):
    monkeypatch.delenv(package_identity.PROGRAM_ENVIRONMENT, raising=False)
    assert package_identity.program_name(argv0) == expected


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
def test_module_spec_preserves_invocation_despite_canonical_source_path(monkeypatch, command):
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", SimpleNamespace(name=f"{command}.__main__"))
    monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, "aisimulate" if command == "aisim" else "aisim")
    assert package_identity.program_name("/site-packages/aisim/__main__.py") == command


def test_worker_processes_inherit_the_invoked_command(monkeypatch):
    worker = "/site-packages/aisim/resource_worker.py"
    for command in package_identity.COMMANDS:
        monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, command)
        assert package_identity.program_name(worker) == command
    monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, "something-else")
    assert package_identity.program_name(worker) == "aisim"


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
@pytest.mark.parametrize("inherited", [None, "opposite"], ids=["unset", "conflicting"])
def test_supervisor_passes_the_actual_command_name_to_its_worker(monkeypatch, tmp_path, command, inherited):
    if inherited is None:
        monkeypatch.delenv(package_identity.PROGRAM_ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, "aisimulate" if command == "aisim" else "aisim")
    monkeypatch.setattr(sys, "argv", [f"/venv/bin/{command}"])
    monkeypatch.setattr(supervision, "discover_host", lambda: SimpleNamespace(process_memory_bytes=0))
    monkeypatch.setattr(supervision, "resolve_budget", lambda _policy, _host: {"memory_limit_bytes": 1})
    seen = {}
    real_popen = subprocess.Popen

    class StopWorker(Exception):
        pass

    def popen(command, *, env, start_new_session):
        seen.update(env)
        # Execute the actual parser in a child with the supervisor's environment
        # and the worker's argv[0], before substituting resource enforcement.
        with monkeypatch.context() as child:
            child.setattr(subprocess, "Popen", real_popen)
            worker = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.argv[0] = '/site-packages/aisim/resource_worker.py'; "
                    "from aisim.cli_args import build_parser; build_parser().parse_args(['unknown-command'])",
                ],
                env={**env, "PYTHONPATH": str(Path(aisim.__file__).resolve().parent.parent)},
                capture_output=True,
                text=True,
                timeout=30,
            )
        assert worker.returncode == 2, worker.stdout + worker.stderr
        seen["worker_error"] = worker.stderr
        raise StopWorker

    monkeypatch.setattr(supervision.subprocess, "Popen", popen)
    with pytest.raises(StopWorker):
        supervision._run_process(
            ["worker"], policy=supervision.ResourceConfig(), timeout=None, events_output=None, directory=tmp_path
        )
    assert seen[package_identity.PROGRAM_ENVIRONMENT] == command
    assert seen["worker_error"].startswith(f"usage: {command} ")
    assert f"{command}: error:" in seen["worker_error"]


def test_every_command_and_option_is_available_under_both_names(monkeypatch):
    signatures = []
    for command in package_identity.COMMANDS:
        monkeypatch.setattr(sys, "argv", [command])
        signatures.append(
            {
                path: [
                    (
                        tuple(a.option_strings),
                        a.dest,
                        a.nargs,
                        a.required,
                        tuple(a.choices) if a.choices is not None else None,
                        a.default,
                    )
                    for a in parser._actions
                ]
                for path, parser in _parsers(build_parser())
            }
        )
    assert signatures[0] == signatures[1]
    assert {(), ("predict",), ("recommend",), ("onboard",), ("perf-model",)} <= signatures[0].keys()


@pytest.mark.parametrize("path", COMMAND_PATHS, ids=lambda p: " ".join(p) or "root")
@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
def test_help_is_available_for_every_command(monkeypatch, capsys, command, path):
    monkeypatch.setattr(sys, "argv", [command])
    with pytest.raises(SystemExit) as exited:
        supervision.main([*path, "--help"])
    assert exited.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(f"usage: {' '.join((command, *path))} ")
    assert captured.err == ""


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
def test_argument_errors_name_the_invoked_command(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", [command])
    with pytest.raises(SystemExit) as exited:
        supervision.main(["unknown-command"])
    assert exited.value.code == 2
    assert f"{command}: error:" in capsys.readouterr().err


def _launch(command, mode, arguments, tmp_path):
    if mode == "console":
        executable = Path(sys.executable).parent / command
        argv = [str(executable), *arguments]
    else:
        argv = [sys.executable, "-m", command, *arguments]
    env = dict(os.environ)
    env.pop(package_identity.PROGRAM_ENVIRONMENT, None)
    # Exercise this checkout in local runs and the exact installed wheel in CI.
    env["PYTHONPATH"] = str(Path(aisim.__file__).resolve().parent.parent)
    return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
@pytest.mark.parametrize("mode", ["console", "module"])
def test_real_launchers_report_the_distribution_version(command, mode, tmp_path):
    result = _launch(command, mode, ["--version"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"aisim {aisim.__version__}\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "path",
    [(), ("predict",), ("recommend",), ("onboard",), ("perf-model",)],
    ids=["root", "predict", "recommend", "onboard", "perf-model"],
)
@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
@pytest.mark.parametrize("mode", ["console", "module"])
def test_real_launchers_preserve_command_identity(command, mode, path, tmp_path):
    result = _launch(command, mode, [*path, "--help"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith(f"usage: {' '.join((command, *path))} ")
    assert result.stderr == ""
