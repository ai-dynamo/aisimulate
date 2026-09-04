# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package contracts for the experimental Sweeper feature."""

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.timeout(30)


def _source_checkout_roots() -> tuple[Path, Path]:
    """Return the package and repository roots for source-only contracts."""
    repo_root = Path(__file__).resolve().parents[2]
    aisimulate_root = repo_root / "python" / "aisimulate"
    source_tree_markers = (
        aisimulate_root / "pyproject.toml",
        repo_root / "crates/core/Cargo.toml",
        repo_root / "Cargo.toml",
    )
    if not all(path.is_file() for path in source_tree_markers):
        pytest.skip("requires the AISimulate source checkout, not only the wheel")
    return aisimulate_root, repo_root


def _ai_dynamo_distribution_or_skip():
    """Return optional Dynamo metadata when cross-repository CI installs it."""
    try:
        return importlib.metadata.distribution("ai-dynamo")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("cross-repository Dynamo contract runs in integration CI")


def test_aisimulate_distribution_publishes_aisimulate_sweeper_package():
    distribution = importlib.metadata.distribution("aisimulate")
    packaged_files = {str(path) for path in distribution.files or ()}

    assert distribution.metadata["Name"] == "aisimulate"
    assert importlib.util.find_spec("aisimulate.replay") is not None
    assert importlib.util.find_spec("aisimulate.sweeper") is not None
    assert importlib.util.find_spec("aisimulate.afd_artifacts") is not None
    assert importlib.util.find_spec("aisimulate.replay.__main__") is None
    assert importlib.util.find_spec("aisimulate.sweeper.__main__") is None
    # Editable installs expose only their .pth/dist-info records. In wheel-based
    # Planner CI, assert the artifact contains the canonical package and no alias.
    if any(path.startswith("aisimulate/") for path in packaged_files):
        assert any(path.startswith("aisimulate/sweeper/") for path in packaged_files)
        assert any(path.startswith("aisimulate/replay/") for path in packaged_files)
        assert not any(path.startswith("aisimulate/spica/") for path in packaged_files)
        assert not any(path.startswith("sweeper/") for path in packaged_files)


def test_aisimulate_native_runtime_imports_from_installed_distribution():
    runtime_spec = importlib.util.find_spec("aisimulate._runtime")

    assert runtime_spec is not None
    runtime = importlib.import_module("aisimulate._runtime")
    assert callable(runtime.run_replay_json)
    assert callable(runtime.run_replay_with_artifacts_json)


def test_aisimulate_exposes_unified_and_aiconfigurator_console_scripts():
    distribution = importlib.metadata.distribution("aisimulate")

    scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts == {
        "aiconfigurator": "aiconfigurator.main:main",
        "aisimulate": "aisimulate.main:main",
    }


def test_ai_dynamo_has_no_aisimulate_extra():
    distribution = _ai_dynamo_distribution_or_skip()

    extras = set(distribution.metadata.get_all("Provides-Extra", []))
    assert {"sweeper", "simulate", "simulation"}.isdisjoint(extras)


def test_aisimulate_has_no_dynamo_or_component_adapter_dependencies():
    distribution = importlib.metadata.distribution("aisimulate")

    requirements = distribution.requires or []
    names = {Requirement(requirement).name.lower() for requirement in requirements}
    assert "ai-dynamo" not in names
    assert "prometheus-api-client" not in names
    assert "filterpy" not in names
    assert "pmdarima" not in names
    assert "prophet" not in names


def test_importing_sweeper_does_not_import_dynamo():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aisimulate.sweeper; "
                "assert not any(name == 'dynamo' or name.startswith('dynamo.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_ai_dynamo_registers_optional_sweeper_providers():
    distribution = _ai_dynamo_distribution_or_skip()
    entry_points = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "aisimulate.sweep_config_providers"
    }

    assert entry_points == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_ai_dynamo_runner_preserves_independent_sla_bounds(
    field: str, bound: float
) -> None:
    _ai_dynamo_distribution_or_skip()
    from dynamo.replay.simulation import DynamoReplayRunner

    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args={},
            num_workers=1,
        ),
        workload={},
        goal={"target": "throughput", "strict_sla": False, "sla": {field: bound}},
    )
    expected = {
        "sla_ttft_ms": None,
        "sla_itl_ms": None,
        "sla_e2e_ms": None,
    }
    expected[f"sla_{field}"] = bound

    assert DynamoReplayRunner._goodput_sla_kwargs(spec) == expected


def test_aisimulate_source_versions_are_synchronized():
    root, repo_root = _source_checkout_roots()
    project = tomllib.loads((root / "pyproject.toml").read_text())
    core = tomllib.loads((repo_root / "crates/core/Cargo.toml").read_text())
    workspace = tomllib.loads((repo_root / "Cargo.toml").read_text())

    expected_python = "0.12.0"
    expected_cargo = "0.12.0"
    assert project["project"]["version"] == expected_python
    assert core["package"]["version"] == expected_cargo
    assert workspace["workspace"]["members"] == ["crates/core"]
    assert project["tool"]["maturin"]["manifest-path"] == "../../crates/core/Cargo.toml"


def test_profiler_does_not_publish_or_reexport_sweeper():
    _ai_dynamo_distribution_or_skip()
    assert importlib.util.find_spec("dynamo.profiler.sweeper") is None
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dynamo.profiler; assert not hasattr(dynamo.profiler, 'sweeper')",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
