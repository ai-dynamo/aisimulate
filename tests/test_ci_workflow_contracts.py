# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    with (WORKFLOW_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_restored_workflows_are_active_at_repository_root() -> None:
    expected = {
        "build-platform-wheels-copied-pr.yml",
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    }

    assert expected.issubset({path.name for path in WORKFLOW_ROOT.glob("*.yml")})


def test_core_ci_selects_migrated_contract_and_parity_suites() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    assert {
        "rust-feature-modes",
        "python-compatibility",
        "rust-python-parity",
        "ci-success",
    }.issubset(jobs)

    application_commands = _run_commands(jobs["application-tests"])
    assert "test_core_public_api.py" in application_commands
    assert "test_single_distribution.py" in application_commands

    parity_commands = _run_commands(jobs["rust-python-parity"])
    assert "test_engine_step_parity.py" in parity_commands
    assert "test_compile_engine_parity.py" in parity_commands

    required_by_aggregate = set(jobs["ci-success"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "rust-python-parity",
    }.issubset(required_by_aggregate)

    required_before_wheel_staging = set(jobs["application-wheel"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "rust-python-parity",
    }.issubset(required_before_wheel_staging)


def test_manual_ci_allows_trusted_wheel_staging_to_be_skipped() -> None:
    aggregate = _workflow("ci.yml")["jobs"]["ci-success"]

    assert aggregate["steps"][0]["env"]["EXPECTED_APPLICATION_WHEEL_RESULT"] == (
        "${{ github.event_name == 'workflow_dispatch' && 'skipped' || 'success' }}"
    )
    assert 'test "${APPLICATION_WHEEL_RESULT}" = "${EXPECTED_APPLICATION_WHEEL_RESULT}"' in _run_commands(aggregate)


def test_privileged_pr_workflows_use_copied_pr_pushes() -> None:
    for filename in (
        "build-platform-wheels-copied-pr.yml",
        "collector-check.yml",
        "prediction-regression-gate.yml",
    ):
        triggers = _workflow(filename)["on"]
        assert "push" in triggers
        assert "pull_request" not in triggers


def test_fpe_generation_uses_the_required_job_container() -> None:
    generate = _workflow("fpe-support-matrix.yml")["jobs"]["generate"]

    assert generate["container"]["image"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"


def test_active_workflows_do_not_call_nested_inert_workflows() -> None:
    for path in WORKFLOW_ROOT.glob("*.yml"):
        assert "./python/aisimulate/.github/workflows/" not in path.read_text(encoding="utf-8")
