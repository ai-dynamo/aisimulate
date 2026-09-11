# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


def _workflow(name: str) -> dict:
    with (WORKFLOW_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _run_resolve_step(script: str, event_name: str, old_ref: str, tmp_path: Path) -> dict[str, str]:
    output_path = tmp_path / f"{event_name}.out"
    output_path.unlink(missing_ok=True)
    subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_OUTPUT": str(output_path),
            "OLD_REF_INPUT": old_ref,
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in output_path.read_text().splitlines())


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
        "readiness",
    }.issubset(jobs)

    assert jobs["platform-wheels"]["uses"] == "./.github/workflows/validate-platform-wheels.yml"
    assert jobs["collector-data"]["uses"] == "./.github/workflows/collector-check.yml"
    assert jobs["prediction-regression"]["uses"] == "./.github/workflows/prediction-regression-gate.yml"

    application_commands = _run_commands(jobs["application-tests"])
    compatibility_commands = _run_commands(jobs["python-compatibility"])
    assert "python/aisimulate/tests/cross_package" in application_commands
    assert "python/aisimulate/tests/cross_package" in compatibility_commands
    assert "test_core_public_api.py" not in application_commands
    assert "test_core_public_api.py" not in compatibility_commands

    recommendation_path = "tests/e2e/cli/test_cli_recommend.py"
    recommendation_steps = [
        step for step in jobs["application-tests"]["steps"] if recommendation_path in step.get("run", "")
    ]
    assert len(recommendation_steps) == 2
    assert "-n auto" not in recommendation_steps[0]["run"]
    assert f"--ignore={recommendation_path}" in recommendation_steps[1]["run"]

    regression = jobs["engine-golden-regression"]
    regression_commands = _run_commands(regression)
    assert regression["name"] == "Engine Golden Regression"
    assert "test_engine_step_parity.py" in regression_commands
    assert "test_compile_engine_parity.py" in regression_commands
    assert regression_commands.count("-c python/aisimulate/pytest.ini") == 2

    feature_mode_commands = _run_commands(jobs["rust-feature-modes"])
    assert "cargo test --workspace --features embed-python,replay-bench" in feature_mode_commands
    assert "--all-features" not in feature_mode_commands
    assert "--no-default-features" not in feature_mode_commands
    assert "PYTHONPATH" not in feature_mode_commands

    required_by_aggregate = set(jobs["readiness"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
    }.issubset(required_by_aggregate)
    aggregate = jobs["readiness"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"

    required_before_wheel_staging = set(jobs["application-wheel"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
    }.issubset(required_before_wheel_staging)


def test_full_ci_aggregate_checks_every_declared_dependency() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate = jobs["readiness"]
    commands = _run_commands(aggregate)

    assert "stage-application-wheel" not in aggregate["needs"]
    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    assert set(aggregate["needs"]) == set(jobs) - {"readiness", "stage-application-wheel"}
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert 'expected = {name: "success" for name in needs}' in commands
    assert 'payload["result"]' in commands
    assert "Full CI did not pass" in commands


def test_fast_ci_owns_static_and_workflow_contract_checks() -> None:
    jobs = _workflow("fast-ci.yml")["jobs"]

    policy_commands = _run_commands(jobs["policy"])
    static_commands = _run_commands(jobs["python-static"])
    assert "tests/test_ci_workflow_contracts.py" in policy_commands
    assert "test_cli_recommend.py" in static_commands
    assert "ruff format --check" in static_commands
    assert "python -m compileall" in static_commands
    assert set(jobs["readiness"]["needs"]) == {
        "policy",
        "python-static",
        "rust-format",
    }
    assert jobs["readiness"]["if"] == "${{ always() }}"


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


def test_full_ci_checkouts_do_not_persist_credentials() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") == "false"


def test_migrated_workflows_keep_reviewed_safety_fixes() -> None:
    collector = _workflow("collector-check.yml")
    prediction = _workflow("prediction-regression-gate.yml")
    platform_wheels = _workflow("validate-platform-wheels.yml")

    collector_commands = _run_commands(collector["jobs"]["check"])
    assert '--no-tags origin "${BASE_SHA}"' in collector_commands
    assert "--depth=1" not in collector_commands
    assert "github.run_id" in collector["concurrency"]["group"]
    assert "cancel-in-progress" not in collector["concurrency"]

    collect_commands = _run_commands(prediction["jobs"]["collect"])
    report_commands = _run_commands(prediction["jobs"]["report"])
    assert "NO_HARNESS" in collect_commands
    assert "pyyaml==6.0.3" in report_commands
    for workflow in (prediction, platform_wheels):
        concurrency_group = workflow["concurrency"]["group"]
        assert "github.event_name == 'push'" in concurrency_group
        assert "startsWith(github.ref, 'refs/heads/pull-request/')" in concurrency_group
        assert "&& github.ref || github.run_id" in concurrency_group
    fast_group = _workflow("fast-ci.yml")["concurrency"]["group"]
    assert "github.event_name == 'pull_request'" in fast_group
    assert "github.event_name == 'push' && startsWith(github.ref, 'refs/heads/pull-request/')" in fast_group
    assert "&& github.ref || github.run_id" in fast_group


def test_migrated_workflows_pin_actions_and_do_not_persist_checkout_credentials() -> None:
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        workflow = _workflow(filename)
        source = (WORKFLOW_ROOT / filename).read_text(encoding="utf-8")
        assert "https://github.com/ai-dynamo/AIConfigurator/tree/" in source

        for uses in re.findall(r"^\s*-?\s*uses:\s+([^\s#]+)", source, re.MULTILINE):
            if uses.startswith("./"):
                continue
            assert re.search(r"@[0-9a-f]{40}$", uses), f"{filename}: {uses} is not pinned"

        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") == "false"


def test_exact_target_checks_bind_expressions_through_step_environments() -> None:
    expression = "${{ inputs.expected_sha || github.sha }}"
    for filename in (
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    ):
        workflow = _workflow(filename)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                assert expression not in step.get("run", "")


def test_platform_wheel_build_and_verifiers_cover_collector_payload() -> None:
    dockerfile = (REPOSITORY_ROOT / "python" / "aisimulate" / "docker" / "Dockerfile").read_text()
    release_verifier = (REPOSITORY_ROOT / "python" / "aisimulate" / "tools" / "verify_release_wheels.py").read_text()
    installed_verifier = (
        REPOSITORY_ROOT / "python" / "aisimulate" / "tools" / "verify_installed_package_layers.py"
    ).read_text()

    assert "COPY python/aisimulate/collector/ /workspace/python/aisimulate/collector/" in dockerfile
    assert "ln -s ../src /workspace/python/aisimulate/aic-core/src" in dockerfile
    assert "test -d /workspace/python/aisimulate/src/aiconfigurator/model_configs" in dockerfile
    assert "test -d /workspace/python/aisimulate/src/aiconfigurator/systems" in dockerfile
    assert '"cases/**/*.yaml"' in release_verifier
    assert '"fpm_forward/**/*.py"' in release_verifier
    assert '"collector/fpm_forward/runtime/fpm_exec.sh"' in installed_verifier
    assert 'importlib.import_module("collector.fpm_forward")' in installed_verifier


def test_prediction_gate_owns_report_dependencies_and_pre_harness_fallback() -> None:
    jobs = _workflow("prediction-regression-gate.yml")["jobs"]
    report_commands = _run_commands(jobs["report"])
    collect_commands = _run_commands(jobs["collect"])

    assert "python -m pip install pyyaml==6.0.3" in report_commands
    assert "NO_HARNESS.txt" in collect_commands
    assert "predates the prediction-regression harness" in collect_commands


def test_prediction_gate_resolves_base_for_push_and_manual_callers(tmp_path: Path) -> None:
    resolve_step = _workflow("prediction-regression-gate.yml")["jobs"]["refs"]["steps"][0]
    script = resolve_step["run"]

    assert _run_resolve_step(script, "push", "main", tmp_path) == {"old": "main"}
    assert _run_resolve_step(script, "push", "", tmp_path) == {"old": "main"}
    assert _run_resolve_step(script, "workflow_dispatch", "release/0.12", tmp_path) == {"old": "release/0.12"}


@pytest.mark.parametrize(
    ("event", "ref", "before", "uses_before"),
    [
        ("push", "refs/heads/main", "a" * 40, True),
        ("push", "refs/heads/release/0.12.0", "b" * 40, True),
        ("push", "refs/heads/pull-request/96", "a" * 40, False),
        ("workflow_dispatch", "refs/heads/main", "a" * 40, False),
        ("workflow_dispatch", "refs/heads/codex/restore-migrated-ci", "", False),
        ("push", "refs/heads/release/0.12.0", "0" * 40, False),
    ],
)
def test_full_ci_comparison_base_uses_previous_lifecycle_commit(
    tmp_path: Path, event: str, ref: str, before: str, uses_before: bool
) -> None:
    result, output = _run_comparison_base(tmp_path, event, ref, before)
    assert result.returncode == 0, result.stderr
    assert output == {"sha": before if uses_before else "c" * 40}
    assert (tmp_path / "gh-called").exists() is not uses_before


@pytest.mark.parametrize(("before", "api_sha", "api_status"), [("bad", "c" * 40, 0), ("", "main", 0), ("", "", 1)])
def test_full_ci_comparison_base_fails_closed(tmp_path: Path, before: str, api_sha: str, api_status: int) -> None:
    result, output = _run_comparison_base(tmp_path, "push", "refs/heads/main", before, api_sha, api_status)
    assert result.returncode != 0
    assert output == {}


def _run_comparison_base(
    tmp_path: Path, event: str, ref: str, before: str, api_sha: str = "c" * 40, api_status: int = 0
) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    jobs = _workflow("ci.yml")["jobs"]
    assert jobs["verify-target"]["outputs"]["comparison-base"] == "${{ steps.comparison-base.outputs.sha }}"
    for name, argument in (("collector-data", "base_sha"), ("prediction-regression", "old-ref")):
        assert "verify-target" in jobs[name]["needs"]
        assert jobs[name]["with"][argument] == "${{ needs.verify-target.outputs.comparison-base }}"
    step = next(step for step in jobs["verify-target"]["steps"] if step.get("id") == "comparison-base")
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\n[[ "$*" == "api -X GET repos/ai-dynamo/aisimulate/commits/main --jq .sha" ]] || exit 9\n'
        'touch "${GH_CALLED}"\nprintf "%s\\n" "${TEST_API_SHA}"\nexit "${TEST_API_STATUS}"\n'
    )
    gh.chmod(0o755)
    output_path = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "BEFORE_SHA": before,
            "REPOSITORY": "ai-dynamo/aisimulate",
            "GITHUB_OUTPUT": str(output_path),
            "GH_CALLED": str(tmp_path / "gh-called"),
            "TEST_API_SHA": api_sha,
            "TEST_API_STATUS": str(api_status),
        },
        capture_output=True,
        text=True,
    )
    output = dict(line.split("=", 1) for line in output_path.read_text().splitlines()) if output_path.exists() else {}
    return result, output


def test_collector_comparison_fetch_preserves_full_history() -> None:
    commands = _run_commands(_workflow("collector-check.yml")["jobs"]["check"])

    assert '--no-tags origin "${BASE_SHA}"' in commands
    assert "--depth=1" not in commands


def test_fpe_job_uses_required_container_without_legacy_lfs_data() -> None:
    generate = _workflow("fpe-support-matrix.yml")["jobs"]["generate"]
    checkout = next(step for step in generate["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "ls-files",
            "python/aisimulate/src/aiconfigurator_core/systems/**/*.txt",
        ],
        text=True,
    ).splitlines()

    assert generate["container"]["image"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"
    assert checkout.get("with", {}).get("lfs") == "true" or not tracked, (
        "FPE checkout must enable LFS before a legacy systems/**/*.txt payload is tracked"
    )


def test_shared_python_rust_setup_is_used_by_same_revision_jobs() -> None:
    action_path = "./.github/actions/setup-python-rust"
    full_ci = _workflow("ci.yml")["jobs"]
    for job_name in (
        "rust",
        "rust-feature-modes",
        "public-api-rust",
        "application-wheel",
        "application-tests",
        "python-compatibility",
        "engine-golden-regression",
        "release-artifact-contract",
    ):
        assert any(step.get("uses") == action_path for step in full_ci[job_name]["steps"])

    assert any(step.get("uses") == action_path for step in _workflow("collector-check.yml")["jobs"]["check"]["steps"])
    assert any(
        step.get("uses") == action_path for step in _workflow("fpe-support-matrix.yml")["jobs"]["generate"]["steps"]
    )


def test_macos_wheel_environment_seeds_pip_for_shared_verification() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)

    commands = _run_commands(action["runs"])
    assert 'uv venv --seed --python 3.13 "${RUNNER_TEMP}/aisimulate-wheel-venv"' in commands
    assert "python -m pip install --quiet wheelhouse/aisimulate-*.whl" in commands


def test_omitted_wheel_base_preserves_dockerfile_default() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)
    dockerfile = (REPOSITORY_ROOT / "python/aisimulate/docker/Dockerfile").read_text()
    docker_default = re.search(r"^ARG WHEEL_BUILD_BASE=(.+)$", dockerfile, re.MULTILINE).group(1)
    wheel_base = action["inputs"]["wheel_base"]
    assert wheel_base["required"] == "false"
    assert wheel_base["default"] == docker_default
    build = next(step for step in action["runs"]["steps"] if step["name"] == "Build Linux wheel")
    omitted_input_arguments = build["with"]["build-args"].replace("${{ inputs.wheel_base }}", wheel_base["default"])
    assert omitted_input_arguments == f"WHEEL_BUILD_BASE={docker_default}"


def test_containerized_workflows_do_not_require_git_lfs_during_checkout() -> None:
    jobs = (
        ("ci.yml", "engine-golden-regression"),
        ("collector-check.yml", "cross-backend-consistency"),
        ("prediction-regression-gate.yml", "collect"),
    )

    for workflow_name, job_name in jobs:
        job = _workflow(workflow_name)["jobs"][job_name]
        checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout.get("with", {}).get("lfs") != "true"


def test_active_workflows_do_not_call_nested_inert_github_assets() -> None:
    for path in WORKFLOW_ROOT.glob("*.yml"):
        assert "./python/aisimulate/.github/" not in path.read_text(encoding="utf-8")
