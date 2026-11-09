# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-identity contracts: aisim/aisimulate overlap warning, command name, and --version."""

from __future__ import annotations

import importlib.metadata
import io
import sys
from types import SimpleNamespace

import pytest

import aisimulate
from aisimulate import package_identity, supervision
from aisimulate.cli_args import build_parser

pytestmark = pytest.mark.unit

LEGACY_RECORD = "aisimulate/__init__.py,sha256=x,1\naisimulate_core/__init__.py,sha256=y,2\n"
SHIM_RECORD = "aisimulate-0.14.0.dist-info/METADATA,sha256=z,3\n"


class _Distribution:
    def __init__(self, name: str, version: str, record: str | None = None) -> None:
        self.name = name
        self.version = version
        self._record = record

    def read_text(self, filename: str) -> str | None:
        assert filename == "RECORD"
        return self._record


def _install(monkeypatch, *distributions: _Distribution) -> None:
    by_name = {distribution.name: distribution for distribution in distributions}

    def lookup(name: str):
        try:
            return by_name[name]
        except KeyError:
            raise importlib.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(importlib.metadata, "distribution", lookup)


AISIM = _Distribution("aisim", "0.14.0", "aisimulate/__init__.py,,\n")
AISIMULATE = _Distribution("aisimulate", "0.13.0", LEGACY_RECORD)


@pytest.mark.parametrize(
    ("record", "owns"),
    [
        ("./aisimulate/__init__.py,sha256=x,1\n", True),
        ("aisimulate\\__init__.py,sha256=x,1\n", True),
        ('"aisimulate_core/a,b.py",sha256=x,1\n', True),
        ("aisimulate-0.13.0.dist-info/RECORD,,\n", False),
        ("aisimulate_extra/__init__.py,sha256=x,1\n", False),
        ("aisimulate,sha256=x,1\n", False),
    ],
    ids=["dot-prefix", "backslash", "quoted-comma", "dist-info", "similar-name", "bare-name"],
)
def test_shared_package_ownership_normalizes_record_paths(record, owns) -> None:
    legacy = _Distribution("aisimulate", "0.13.0", record)
    assert package_identity._owns_shared_packages(legacy) is owns


@pytest.mark.parametrize(
    "installed",
    [(), (AISIM,), (AISIMULATE,)],
    ids=["none", "aisim-only", "aisimulate-only"],
)
def test_no_overlap_message_unless_both_distributions_are_installed(monkeypatch, installed) -> None:
    _install(monkeypatch, *installed)
    assert package_identity.overlap_message() is None


def test_overlap_message_gives_the_safe_repair_sequence(monkeypatch) -> None:
    _install(monkeypatch, AISIM, AISIMULATE)

    message = package_identity.overlap_message()

    assert message is not None
    # Same lead sentence as scripts/release/package_identity.py.
    assert message.startswith("aisim and aisimulate are both installed and own overlapping files.")
    assert "aisimulate 0.13.0 is the pre-0.14 name of aisim 0.14.0" in message
    assert "Do not run `pip uninstall aisimulate` on its own" in message
    assert "pip uninstall -y aisimulate && pip install --force-reinstall --no-deps aisim==0.14.0" in message
    assert "uv pip uninstall aisimulate && uv pip install --reinstall --no-deps aisim==0.14.0" in message


def test_overlap_without_record_assumes_the_historical_layout(monkeypatch) -> None:
    _install(monkeypatch, AISIM, _Distribution("aisimulate", "0.13.0", None))
    assert package_identity.overlap_message() is not None


def test_dependency_only_legacy_shim_does_not_warn(monkeypatch) -> None:
    _install(monkeypatch, AISIM, _Distribution("aisimulate", "0.14.0", SHIM_RECORD))
    assert package_identity.overlap_message() is None


def test_unreadable_metadata_never_blocks_the_cli(monkeypatch) -> None:
    def broken(name: str):
        raise OSError(f"cannot read {name} metadata")

    monkeypatch.setattr(importlib.metadata, "distribution", broken)
    assert package_identity.overlap_message() is None

    class _NoVersion(_Distribution):
        @property
        def version(self) -> str:
            raise KeyError("Version")

        @version.setter
        def version(self, value: str) -> None:
            pass

    _install(monkeypatch, _NoVersion("aisim", "ignored"), AISIMULATE)
    assert package_identity.overlap_message() is None


def test_warn_on_overlap_writes_one_prefixed_warning(monkeypatch) -> None:
    _install(monkeypatch, AISIM, AISIMULATE)
    monkeypatch.setattr(sys, "argv", ["/venv/bin/aisim"])
    stream = io.StringIO()

    assert package_identity.warn_on_overlap(stream) is True

    output = stream.getvalue()
    assert output.startswith("aisim: warning: aisim and aisimulate are both installed")
    assert output.count("warning:") == 1


def test_warn_on_overlap_is_silent_for_a_clean_install(monkeypatch) -> None:
    _install(monkeypatch, AISIM)
    stream = io.StringIO()
    assert package_identity.warn_on_overlap(stream) is False
    assert stream.getvalue() == ""


def test_cli_warns_before_parsing_so_help_carries_the_repair(monkeypatch, capsys) -> None:
    _install(monkeypatch, AISIM, AISIMULATE)
    monkeypatch.setattr(sys, "argv", ["/venv/bin/aisimulate"])

    with pytest.raises(SystemExit) as exited:
        supervision.main(["--help"])

    assert exited.value.code == 0
    captured = capsys.readouterr()
    assert captured.err.startswith("aisimulate: warning: aisim and aisimulate are both installed")
    assert captured.out.startswith("usage: aisimulate ")


def test_cli_is_silent_for_a_clean_install(monkeypatch, capsys) -> None:
    _install(monkeypatch, AISIM)
    with pytest.raises(SystemExit):
        supervision.main(["--help"])
    assert "warning" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/venv/bin/aisim", "aisim"),
        ("/venv/bin/aisimulate", "aisimulate"),
        ("/venv/Scripts/aisim.exe", "aisim"),
        ("/venv/bin/aisim-script.py", "aisim"),
        ("/venv/lib/python3.12/site-packages/aisimulate/__main__.py", "aisimulate"),
        ("", "aisimulate"),
    ],
)
def test_program_name_follows_the_invoked_command(monkeypatch, argv0, expected) -> None:
    monkeypatch.delenv(package_identity.PROGRAM_ENVIRONMENT, raising=False)
    assert package_identity.program_name(argv0) == expected


def test_worker_processes_inherit_the_invoked_command(monkeypatch) -> None:
    worker = "/venv/lib/python3.12/site-packages/aisimulate/resource_worker.py"
    monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, "aisim")
    assert package_identity.program_name(worker) == "aisim"
    monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, "something-else")
    assert package_identity.program_name(worker) == "aisimulate"


@pytest.mark.parametrize("inherited", [None, "aisimulate"], ids=["unset", "stale"])
def test_supervisor_passes_the_command_name_to_its_worker(monkeypatch, tmp_path, inherited) -> None:
    if inherited is None:
        monkeypatch.delenv(package_identity.PROGRAM_ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(package_identity.PROGRAM_ENVIRONMENT, inherited)
    monkeypatch.setattr(sys, "argv", ["/venv/bin/aisim"])
    monkeypatch.setattr(supervision, "discover_host", lambda: SimpleNamespace(process_memory_bytes=0))
    monkeypatch.setattr(supervision, "resolve_budget", lambda _policy, _host: {"memory_limit_bytes": 1})
    seen = {}

    class _Stop(Exception):
        pass

    def popen(command, *, env, start_new_session):
        seen.update(env)
        raise _Stop

    monkeypatch.setattr(supervision.subprocess, "Popen", popen)
    with pytest.raises(_Stop):
        supervision._run_process(
            ["worker"], policy=supervision.ResourceConfig(), timeout=None, events_output=None, directory=tmp_path
        )
    assert seen[package_identity.PROGRAM_ENVIRONMENT] == "aisim"


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
def test_help_usage_names_the_invoked_command(monkeypatch, command) -> None:
    monkeypatch.setattr(sys, "argv", [f"/venv/bin/{command}"])
    parser = build_parser()
    assert parser.format_usage().startswith(f"usage: {command} ")
    onboard = parser._subparsers._group_actions[0].choices["onboard"]
    assert onboard.prog == f"{command} onboard"


@pytest.mark.parametrize("command", ["aisim", "aisimulate"])
def test_version_names_the_distribution_for_both_commands(monkeypatch, capsys, command) -> None:
    _install(monkeypatch, AISIM)
    monkeypatch.setattr(sys, "argv", [f"/venv/bin/{command}"])

    with pytest.raises(SystemExit) as exited:
        supervision.main(["--version"])

    assert exited.value.code == 0
    assert capsys.readouterr().out == f"aisim {aisimulate.__version__}\n"
