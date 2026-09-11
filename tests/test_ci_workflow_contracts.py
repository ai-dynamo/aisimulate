# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.check_application_test_inventory import Inventory, assignment
from scripts.select_full_ci import COMPONENTS, select_components

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


@pytest.mark.parametrize(
    ("path", "markers", "expected"),
    [
        ("tests/unit/generator/test_unmarked.py", set(), "unit"),
        ("tests/unit/sdk/test_mixed.py", {"integration"}, "unit"),
        ("tests/golden/generator/test_contract.py", set(), "unit"),
        ("tests/integration/test_new.py", set(), "integration"),
        ("tests/cross_package/test_public_api.py", set(), "contracts"),
        ("tests/e2e/cli/test_new.py", {"build"}, "cli-build"),
        ("tests/e2e/support_matrix/test_new.py", {"build"}, "support-matrix"),
        ("tests/e2e/tools/test_new.py", {"build"}, "tools-build"),
    ],
)
def test_collected_inventory_assigns_cases_without_requiring_unit_markers(path, markers, expected):
    assert assignment(path, markers, {})[0] == expected


def test_collected_inventory_requires_explicit_manual_exceptions():
    path = "tests/e2e/cli/test_full_sweep.py"
    with pytest.raises(ValueError, match="unassigned collected test"):
        assignment(path, {"e2e", "sweep"}, {})
    assert assignment(path, {"e2e", "sweep"}, {path: "manual compatibility sweep"}) == (
        "manual",
        "manual compatibility sweep",
    )
    with pytest.raises(ValueError, match="unassigned collected test"):
        assignment("tests/new_category/test_new.py", {"unit"}, {})


def test_collected_inventory_reports_unexpected_skips_and_collection_failures():
    inventory = Inventory(
        {"manual_suites": {}, "optional_collection_skips": {"tests/unit/test_tensor.py": "real torch"}}
    )
    inventory.pytest_collectreport(
        SimpleNamespace(failed=False, skipped=True, nodeid="tests/unit/test_tensor.py", longrepr="real torch required")
    )
    assert not inventory.errors
    inventory.pytest_collectreport(
        SimpleNamespace(
            failed=False,
            skipped=True,
            nodeid="tests/integration/test_native.py",
            longrepr="native extension unavailable",
        )
    )
    inventory.pytest_collectreport(
        SimpleNamespace(failed=True, skipped=False, nodeid="tests/unit/test_bad.py", longrepr="import failed")
    )
    assert len(inventory.errors) == 2
    assert "unexpected collection skip" in inventory.errors[0]
    assert "collection failed" in inventory.errors[1]


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


def test_platform_wheels_require_the_installed_fpe_exercise() -> None:
    action = yaml.safe_load((ACTION_ROOT / "build-platform-wheel" / "action.yml").read_text())
    verification = next(
        step for step in action["runs"]["steps"] if step.get("name") == "Verify installed unified wheel"
    )
    assert "if" not in verification
    assert "--exercise-engine --exercise-fpe" in verification["run"]


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

    application_test_wheel = jobs["application-test-wheel"]
    assert application_test_wheel["timeout-minutes"] == "10"
    assert {"fast-ci", "select-full-ci"}.issubset(application_test_wheel["needs"])
    assert "application-test-wheel" in jobs["application-tests"]["needs"]
    assert {shard["suite"] for shard in jobs["application-tests"]["strategy"]["matrix"]["shard"]} == {
        "contracts",
        "unit",
        "integration",
        "cli-build",
        "support-matrix",
        "tools-build",
    }
    application_commands = _run_commands(jobs["application-tests"])
    compatibility_commands = _run_commands(jobs["python-compatibility"])
    assert "python/aisimulate/tests/cross_package" in application_commands
    assert "python/aisimulate/tests/cross_package" in compatibility_commands
    assert "tests/unit tests/golden" in application_commands
    integration_steps = [
        step for step in jobs["application-tests"]["steps"] if step.get("if") == "matrix.shard.suite == 'integration'"
    ]
    assert len(integration_steps) == 1
    assert "tests/integration" in integration_steps[0]["run"]
    assert "--ignore" not in integration_steps[0]["run"]
    assert "-m" not in shlex.split(integration_steps[0]["run"])[3:]
    assert integration_steps[0]["working-directory"] == "python/aisimulate"
    assert "scripts/check_application_test_inventory.py" in application_commands
    assert "tests/e2e/cli" in application_commands
    assert "tests/e2e/support_matrix" in application_commands
    assert "tests/e2e/tools" in application_commands
    assert "test_core_public_api.py" not in application_commands
    assert "test_core_public_api.py" not in compatibility_commands

    recommendation_path = "tests/e2e/cli/test_cli_recommend.py"
    recommendation_steps = [
        step for step in jobs["application-tests"]["steps"] if recommendation_path in step.get("run", "")
    ]
    assert len(recommendation_steps) == 2
    assert "-n auto" not in recommendation_steps[0]["run"]
    assert f"--ignore={recommendation_path}" in recommendation_steps[1]["run"]

    build_marker_files = {
        path.relative_to(REPOSITORY_ROOT / "python" / "aisimulate").as_posix()
        for path in (REPOSITORY_ROOT / "python" / "aisimulate" / "tests").rglob("test_*.py")
        if "pytest.mark.build" in path.read_text(encoding="utf-8")
    }
    assert build_marker_files
    assert all(
        path.startswith(("tests/e2e/cli/", "tests/e2e/support_matrix/", "tests/e2e/tools/"))
        for path in build_marker_files
    )

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
        "application-test-wheel",
    }.issubset(required_by_aggregate)
    aggregate = jobs["readiness"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"

    required_before_wheel_staging = set(jobs["stage-application-wheel"]["needs"]) | set(jobs["readiness"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "application-wheel",
    }.issubset(required_before_wheel_staging)
    application_wheel_commands = _run_commands(jobs["application-wheel"])
    assert "maturin build" not in application_wheel_commands
    assert any(
        step.get("uses", "").startswith("dtolnay/rust-toolchain@") for step in jobs["application-wheel"]["steps"]
    )
    assert any(
        step.get("with", {}).get("name") == "application-test-wheel-${{ matrix.arch }}"
        for step in jobs["application-wheel"]["steps"]
    )


def test_full_ci_aggregate_checks_every_declared_dependency() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate = jobs["readiness"]
    commands = _run_commands(aggregate)

    assert "select-full-ci" in aggregate["needs"]
    assert "stage-application-wheel" not in aggregate["needs"]
    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    assert set(aggregate["needs"]) == set(jobs) - {"readiness", "stage-application-wheel"}
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert aggregate["steps"][0]["env"]["PLAN_JSON"] == ("${{ toJSON(needs.select-full-ci.outputs) }}")
    assert 'selected not in {"true", "false"}' in commands
    assert '"success" if selected == "true" else "skipped"' in commands
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
        "application-test-wheel",
        "python-compatibility",
        "engine-golden-regression",
        "release-artifact-contract",
    ):
        assert any(step.get("uses") == action_path for step in full_ci[job_name]["steps"])

    assert any(step.get("uses") == action_path for step in _workflow("collector-check.yml")["jobs"]["check"]["steps"])
    assert any(
        step.get("uses") == action_path
        for step in _workflow("fpe-support-matrix.yml")["jobs"]["prepare-wheel"]["steps"]
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


def _run_full_ci_aggregate(
    results: dict[str, str],
    plan: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate = jobs["readiness"]
    script = aggregate["steps"][0]["run"]
    needs = {name: {"result": result, "outputs": {}} for name, result in results.items()}
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "NEEDS_JSON": json.dumps(needs),
            "PLAN_JSON": json.dumps(plan),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _run_workflow_script(
    script: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_selective_full_ci_keeps_the_aggregate_fail_closed() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate_needs = set(jobs["readiness"]["needs"])
    component_jobs = {component.replace("_", "-") for component in COMPONENTS}
    assert component_jobs.issubset(aggregate_needs)

    for component in COMPONENTS:
        job = jobs[component.replace("_", "-")]
        assert "select-full-ci" in job["needs"]
        assert f"needs.select-full-ci.outputs.{component} == 'true'" in job["if"]

    wheel_condition = jobs["application-wheel"]["if"]
    assert "needs.fast-ci.result == 'success'" in wheel_condition
    assert "needs.select-full-ci.result == 'success'" in wheel_condition


def test_full_ci_selector_uses_the_complete_exact_head_pr_change_set() -> None:
    selector = _workflow("ci.yml")["jobs"]["select-full-ci"]
    commands = _run_commands(selector)

    assert "pulls/${pr_number}/files?per_page=100" in commands
    assert "--paginate" in commands
    assert ".previous_filename" in commands
    assert "@base64" in commands
    assert "changed_files > 3000" in commands
    assert '"${pr_head}" != "${GITHUB_SHA}"' in commands
    assert "workflow_dispatch:*|push:refs/heads/main|push:refs/heads/release/*" in commands
    assert "force_all=true" in commands


def test_full_ci_scope_resolver_handles_copy_manual_and_race_cases(
    tmp_path: Path,
) -> None:
    selector = _workflow("ci.yml")["jobs"]["select-full-ci"]
    scope_script = next(
        step["run"] for step in selector["steps"] if step.get("name") == "Resolve the trusted change set"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ ${FAKE_GH_EXIT:-0} != 0 ]]; then exit "${FAKE_GH_EXIT}"; fi\n'
        "args=$*\n"
        "if [[ ${args} == *'.changed_files'* ]]; then\n"
        "  printf '%s\\n' \"${FAKE_CHANGED_FILES:-2}\"\n"
        "elif [[ ${args} == *'/files?per_page=100'* ]]; then\n"
        "  if [[ ${FAKE_FAIL_FILES:-false} == true ]]; then exit 1; fi\n"
        "  printf '%s\\n' \"${FAKE_FILES:-README.md}\"\n"
        "elif [[ ${args} == *'.head.sha'* ]]; then\n"
        "  printf '%s\\n' \"${FAKE_PR_HEAD:-}\"\n"
        "else\n"
        "  exit 2\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    output = tmp_path / "copy-output"
    target_sha = "0123456789abcdef"
    copy_env = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/pull-request/136",
        "GITHUB_SHA": target_sha,
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(tmp_path),
        "REPOSITORY": "ai-dynamo/aisimulate",
        "FAKE_PR_HEAD": target_sha,
        "FAKE_FILES": "\n".join(
            base64.b64encode(path.encode()).decode() for path in ("README.md", "crates/core/src/replay/event.rs")
        ),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    copy_result = _run_workflow_script(scope_script, copy_env)
    assert copy_result.returncode == 0, copy_result.stdout + copy_result.stderr
    assert "force_all=false" in output.read_text(encoding="utf-8")
    encoded_paths = (tmp_path / "full-ci-paths.b64").read_text(encoding="ascii").splitlines()
    assert [base64.b64decode(path).decode() for path in encoded_paths] == [
        "README.md",
        "crates/core/src/replay/event.rs",
    ]

    output.unlink()
    truncated = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_FAIL_FILES": "true"},
    )
    assert truncated.returncode != 0

    oversized = _run_workflow_script(
        scope_script,
        {
            **copy_env,
            "FAKE_CHANGED_FILES": "3001",
            "FAKE_FAIL_FILES": "true",
        },
    )
    assert oversized.returncode == 0, oversized.stdout + oversized.stderr
    assert "force_all=true" in output.read_text(encoding="utf-8")

    raced = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_PR_HEAD": "fedcba9876543210"},
    )
    assert raced.returncode != 0
    assert "PR head changed" in raced.stdout

    api_failure = _run_workflow_script(
        scope_script,
        {**copy_env, "FAKE_GH_EXIT": "1"},
    )
    assert api_failure.returncode != 0

    invalid_ref = _run_workflow_script(
        scope_script,
        {**copy_env, "GITHUB_REF": "refs/heads/pull-request/not-a-number"},
    )
    assert invalid_ref.returncode != 0
    assert "invalid trusted PR copy ref" in invalid_ref.stdout

    manual_output = tmp_path / "manual-output"
    manual = _run_workflow_script(
        scope_script,
        {
            **copy_env,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/codex/topic",
            "GITHUB_OUTPUT": str(manual_output),
            "FAKE_GH_EXIT": "1",
        },
    )
    assert manual.returncode == 0, manual.stdout + manual.stderr
    assert "force_all=true" in manual_output.read_text(encoding="utf-8")


def test_full_ci_aggregate_accepts_only_explicit_na_results() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    plan["application_tests"] = "true"
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        **{
            component.replace("_", "-"): ("success" if plan[component] == "true" else "skipped")
            for component in COMPONENTS
        },
        "application-test-wheel": "success",
    }

    passed = _run_full_ci_aggregate(results, plan)
    assert passed.returncode == 0, passed.stdout + passed.stderr

    results["application-tests"] = "skipped"
    missing_selected_job = _run_full_ci_aggregate(results, plan)
    assert missing_selected_job.returncode != 0
    assert "application-tests=skipped, expected success" in missing_selected_job.stdout

    results["application-tests"] = "success"
    results["rust"] = "success"
    unexpected_unselected_job = _run_full_ci_aggregate(results, plan)
    assert unexpected_unselected_job.returncode != 0
    assert "rust=success, expected skipped" in unexpected_unselected_job.stdout


def test_full_ci_aggregate_rejects_missing_selection_output() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    del plan["collector_data"]
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        **{component.replace("_", "-"): "skipped" for component in COMPONENTS},
        "application-test-wheel": "skipped",
    }

    result = _run_full_ci_aggregate(results, plan)
    assert result.returncode != 0
    assert "invalid selection outputs: collector_data=None" in result.stdout


def test_full_ci_aggregate_rejects_missing_dependency() -> None:
    plan = dict.fromkeys(COMPONENTS, "false")
    results = {
        "verify-target": "success",
        "select-full-ci": "success",
        "fast-ci": "success",
        **{component.replace("_", "-"): "skipped" for component in COMPONENTS},
        "application-test-wheel": "skipped",
    }
    del results["collector-data"]

    result = _run_full_ci_aggregate(results, plan)
    assert result.returncode != 0
    assert "missing dependencies: collector-data" in result.stdout


def test_full_ci_selector_skips_heavy_jobs_for_documentation() -> None:
    plan = select_components(["README.md", "docs/architecture.md"])

    assert plan["run_all"] is False
    assert not any(plan["components"].values())


def test_full_ci_selector_maps_python_rust_and_data_boundaries() -> None:
    python_plan = select_components(["python/aisimulate/src/aisimulate/traffic.py"])
    assert {component for component, selected in python_plan["components"].items() if selected} == {
        "platform_wheels",
        "application_wheel",
        "application_tests",
        "python_compatibility",
        "release_artifact_contract",
    }

    rust_plan = select_components(["crates/core/src/replay/event.rs"])
    assert rust_plan["components"]["rust"] is True
    assert rust_plan["components"]["prediction_regression"] is True
    assert rust_plan["components"]["collector_data"] is True
    assert rust_plan["components"]["cargo_deny"] is False

    data_plan = select_components(["python/aisimulate/src/aiconfigurator_core/systems/data/b200/op.parquet"])
    assert data_plan["components"]["collector_data"] is True
    assert data_plan["components"]["prediction_regression"] is True
    assert data_plan["components"]["engine_golden_regression"] is True


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["future/unclassified.file"],
        [".github/workflows/ci.yml"],
        ["scripts/select_full_ci.py"],
    ],
)
def test_full_ci_selector_defaults_unknown_or_contract_changes_to_all(
    paths: list[str],
) -> None:
    plan = select_components(paths)

    assert plan["run_all"] is True
    assert all(plan["components"].values())


def test_full_ci_selector_force_all_and_path_validation() -> None:
    forced = select_components(["README.md"], force_all=True)
    assert forced["run_all"] is True
    assert all(forced["components"].values())

    with pytest.raises(ValueError, match="repository-relative"):
        select_components(["../outside.py"])

    whitespace_name = select_components([" README.md"])
    assert whitespace_name["run_all"] is True
    backslash_name = select_components([r"docs\architecture.md"])
    assert backslash_name["run_all"] is True
    with pytest.raises(ValueError, match="repository-relative"):
        select_components(["docs/readme.md\nREADME.md"])


def test_full_ci_selector_cli_decodes_paths_and_writes_outputs(tmp_path: Path) -> None:
    encoded = tmp_path / "paths.b64"
    encoded.write_text(
        base64.b64encode(b"README.md").decode() + "\n",
        encoding="ascii",
    )
    output = tmp_path / "github-output"
    summary = tmp_path / "summary.md"

    result = subprocess.run(
        [
            "python3",
            "scripts/select_full_ci.py",
            "--base64-paths-file",
            str(encoded),
            "--github-output",
            str(output),
            "--summary",
            str(summary),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_all=false" in output.read_text(encoding="utf-8")
    assert "application_tests=false" in output.read_text(encoding="utf-8")
    assert "Explicitly N/A" in summary.read_text(encoding="utf-8")


def test_python_dependency_changes_run_the_complete_matrix() -> None:
    for path in (
        "python/aisimulate/pyproject.toml",
        "python/aisimulate/uv.lock",
        "python/aisimulate/pytest.ini",
    ):
        plan = select_components([path])

        assert plan["run_all"] is True
        assert all(plan["components"].values())


def test_full_ci_selector_matches_the_independent_mapping_oracle() -> None:
    oracle_path = REPOSITORY_ROOT / ".github" / "full-ci-selection-cases.yml"
    oracle = yaml.safe_load(oracle_path.read_text(encoding="utf-8"))
    assert oracle["schema_version"] == 1
    cases = oracle["cases"]
    assert len({case["id"] for case in cases}) == len(cases)

    observed_components: set[str] = set()
    for case in cases:
        plan = select_components(
            case["paths"],
            force_all=case.get("force_all", False),
        )
        actual = {component for component, is_selected in plan["components"].items() if is_selected}
        expected = set(COMPONENTS) if case["run_all"] else set(case["selected"])
        assert plan["run_all"] is case["run_all"], case["id"]
        assert actual == expected, case["id"]
        if not case["run_all"]:
            observed_components.update(actual)

    assert observed_components == set(COMPONENTS)


def test_parallel_test_matrix_has_no_missing_or_duplicate_partitions():
    jobs = _workflow("ci.yml")["jobs"]
    shards = jobs["application-tests"]["strategy"]["matrix"]["shard"]
    for suite in {entry["suite"] for entry in shards}:
        entries = [entry for entry in shards if entry["suite"] == suite]
        counts = {int(entry["groups"]) for entry in entries}
        assert len(counts) == 1
        assert sorted(int(entry["group"]) for entry in entries) == list(range(1, counts.pop() + 1))
    assert "application-tests" not in jobs["application-wheel"]["needs"]
    assert {
        "application-tests",
        "application-wheel",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
    }.issubset(set(jobs["stage-application-wheel"]["needs"]) | set(jobs["readiness"]["needs"]))
    commands = _run_commands(jobs["application-tests"])
    assert commands.count("--splitting-algorithm least_duration") == 2
    assert commands.count("--splits ${{ matrix.shard.groups }} --group ${{ matrix.shard.group }}") == 2


@pytest.mark.parametrize("failed", ["package", "rust", "neither"])
def test_parallel_native_preparation_propagates_either_build_failure(tmp_path, failed):
    job = _workflow("ci.yml")["jobs"]["rust-feature-modes"]
    script = next(step["run"] for step in job["steps"] if step.get("name", "").endswith("concurrently"))
    for executable, component in (("python", "package"), ("cargo", "rust")):
        path = tmp_path / executable
        path.write_text(
            f"#!/bin/sh\necho {component} >> '{tmp_path}/completed'\nexit {1 if failed == component else 0}\n"
        )
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "pythonLocation": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (failed == "neither")
    assert set((tmp_path / "completed").read_text().splitlines()) == {"package", "rust"}
