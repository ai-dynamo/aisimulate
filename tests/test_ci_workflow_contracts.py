# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


def _workflow(name: str) -> dict:
    with (WORKFLOW_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_restored_workflows_are_active_at_repository_root() -> None:
    expected = {
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    }

    assert expected.issubset({path.name for path in WORKFLOW_ROOT.glob("*.yml")})
    assert not (WORKFLOW_ROOT / "build-platform-wheels-copied-pr.yml").exists()


def test_full_ci_owns_migrated_expensive_suites() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
        "full-ci-success",
    }.issubset(jobs)

    assert jobs["platform-wheels"]["uses"] == "./.github/workflows/validate-platform-wheels.yml"
    assert jobs["collector-data"]["uses"] == "./.github/workflows/collector-check.yml"
    assert jobs["prediction-regression"]["uses"] == "./.github/workflows/prediction-regression-gate.yml"

    application_commands = _run_commands(jobs["application-tests"])
    assert "test_core_public_api.py" in application_commands
    assert "test_single_distribution.py" in application_commands

    golden_commands = _run_commands(jobs["engine-golden-regression"])
    assert "test_engine_step_parity.py" in golden_commands
    assert "test_compile_engine_parity.py" in golden_commands
    assert golden_commands.count("-c python/aisimulate/pytest.ini") == 2

    feature_mode_commands = _run_commands(jobs["rust-feature-modes"])
    assert "cargo test --workspace --features embed-python,replay-bench" in feature_mode_commands
    assert "--all-features" not in feature_mode_commands
    assert "PYTHONPATH" not in feature_mode_commands

    required_by_aggregate = set(jobs["full-ci-success"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
    }.issubset(required_by_aggregate)
    aggregate = jobs["full-ci-success"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"

    required_before_wheel_staging = set(jobs["application-wheel"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
    }.issubset(required_before_wheel_staging)


def test_fast_ci_owns_static_and_workflow_contract_checks() -> None:
    jobs = _workflow("fast-ci.yml")["jobs"]

    policy_commands = _run_commands(jobs["policy"])
    static_commands = _run_commands(jobs["python-static"])
    assert "tests/test_ci_workflow_contracts.py" in policy_commands
    assert "test_cli_recommend.py" in static_commands
    assert "ruff format --check" in static_commands
    assert "python -m compileall" in static_commands
    assert set(jobs["fast-ci-success"]["needs"]) == {
        "policy",
        "python-static",
        "rust-format",
    }
    assert jobs["fast-ci-success"]["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"


def test_expensive_workflows_are_reusable_full_ci_components() -> None:
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        triggers = _workflow(filename)["on"]
        assert "workflow_call" in triggers
        assert "workflow_dispatch" in triggers
        assert "push" not in triggers
        assert "pull_request" not in triggers


def test_full_ci_propagates_the_exact_sha_to_reusable_gates() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    for job_name in ("platform-wheels", "collector-data", "prediction-regression"):
        assert jobs[job_name]["with"]["expected_sha"] == "${{ github.sha }}"

    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        inputs = _workflow(filename)["on"]["workflow_call"]["inputs"]
        assert "expected_sha" in inputs


def test_migrated_workflows_keep_reviewed_safety_fixes() -> None:
    collector = _workflow("collector-check.yml")
    prediction = _workflow("prediction-regression-gate.yml")

    collector_commands = _run_commands(collector["jobs"]["check"])
    assert "git fetch --no-tags origin" in collector_commands
    assert "--depth=1" not in collector_commands
    assert "github.run_id" in collector["concurrency"]["group"]

    collect_commands = _run_commands(prediction["jobs"]["collect"])
    report_commands = _run_commands(prediction["jobs"]["report"])
    assert "NO_HARNESS" in collect_commands
    assert "pyyaml==6.0.3" in report_commands
    assert "github.run_id" in prediction["concurrency"]["group"]


def test_platform_wheel_build_includes_and_verifies_collector_payload() -> None:
    dockerfile = (REPOSITORY_ROOT / "python/aisimulate/docker/Dockerfile").read_text(encoding="utf-8")
    verifier = (REPOSITORY_ROOT / "python/aisimulate/tools/verify_installed_package_layers.py").read_text(
        encoding="utf-8"
    )

    assert "COPY python/aisimulate/collector/ /workspace/python/aisimulate/collector/" in dockerfile
    assert '"collector/model_cases.py"' in verifier
    assert '"collector/fpm_forward/runtime/fpm_exec.sh"' in verifier


def test_fpe_generation_uses_the_required_job_container() -> None:
    generate = _workflow("fpe-support-matrix.yml")["jobs"]["generate"]

    assert generate["container"]["image"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"


def test_macos_wheel_environment_seeds_pip_for_shared_verification() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)

    commands = _run_commands(action["runs"])
    assert 'uv venv --seed --python 3.13 "${RUNNER_TEMP}/aisimulate-wheel-venv"' in commands
    assert "python -m pip install --quiet wheelhouse/aisimulate-*.whl" in commands


def test_fpe_checkout_matches_tracked_lfs_payload() -> None:
    generate = _workflow("fpe-support-matrix.yml")["jobs"]["generate"]
    checkout = next(step for step in generate["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    tracked = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "python/aisimulate/src/aiconfigurator_core/systems/**/*.txt"],
        text=True,
    ).splitlines()

    assert checkout.get("with", {}).get("lfs") == "true" or not tracked, (
        "FPE checkout must enable LFS before a legacy systems/**/*.txt payload is tracked"
    )


def test_active_workflows_do_not_call_nested_inert_workflows() -> None:
    for path in WORKFLOW_ROOT.glob("*.yml"):
        assert "./python/aisimulate/.github/" not in path.read_text(encoding="utf-8")
