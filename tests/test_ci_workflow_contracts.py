# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.select_full_ci import COMPONENTS, select_components

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


def _workflow(name: str) -> dict:
    with (WORKFLOW_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _run_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _run_full_ci_aggregate(
    results: dict[str, str],
    plan: dict[str, str],
    *,
    expected_stage: str = "skipped",
) -> subprocess.CompletedProcess[str]:
    aggregate = _workflow("ci.yml")["jobs"]["full-ci-success"]
    script = aggregate["steps"][0]["run"]
    needs = {name: {"result": result, "outputs": {}} for name, result in results.items()}
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "NEEDS_JSON": json.dumps(needs),
            "PLAN_JSON": json.dumps(plan),
            "EXPECTED_STAGE_RESULT": expected_stage,
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

    application_test_wheel = jobs["application-test-wheel"]
    assert application_test_wheel["timeout-minutes"] == "10"
    assert {"fast-ci", "select-full-ci"}.issubset(application_test_wheel["needs"])
    assert "application-test-wheel" in jobs["application-tests"]["needs"]
    assert set(jobs["application-tests"]["strategy"]["matrix"]["shard"]) == {
        "contracts",
        "unit",
        "cli-build",
        "support-matrix",
        "tools-build",
    }
    application_commands = _run_commands(jobs["application-tests"])
    compatibility_commands = _run_commands(jobs["python-compatibility"])
    assert "python/aisimulate/tests/cross_package" in application_commands
    assert "python/aisimulate/tests/cross_package" in compatibility_commands
    assert "-m 'unit and not build'" in application_commands
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

    required_by_aggregate = set(jobs["full-ci-success"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
        "application-test-wheel",
    }.issubset(required_by_aggregate)
    aggregate = jobs["full-ci-success"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"

    required_before_wheel_staging = set(jobs["application-wheel"]["needs"])
    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "application-test-wheel",
    }.issubset(required_before_wheel_staging)
    application_wheel_commands = _run_commands(jobs["application-wheel"])
    assert "maturin build" not in application_wheel_commands
    assert any(
        step.get("with", {}).get("name") == "application-test-wheel-${{ matrix.arch }}"
        for step in jobs["application-wheel"]["steps"]
    )


def test_full_ci_aggregate_checks_every_declared_dependency() -> None:
    aggregate = _workflow("ci.yml")["jobs"]["full-ci-success"]
    commands = _run_commands(aggregate)

    assert "select-full-ci" in aggregate["needs"]
    assert "stage-application-wheel" in aggregate["needs"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert aggregate["steps"][0]["env"]["PLAN_JSON"] == ("${{ toJSON(needs.select-full-ci.outputs) }}")
    assert 'selected not in {"true", "false"}' in commands
    assert '"success" if selected == "true" else "skipped"' in commands
    assert 'expected["stage-application-wheel"] = os.environ["EXPECTED_STAGE_RESULT"]' in commands
    assert 'payload["result"]' in commands
    assert "Full CI did not pass" in commands


def test_selective_full_ci_keeps_the_aggregate_fail_closed() -> None:
    jobs = _workflow("ci.yml")["jobs"]
    aggregate_needs = set(jobs["full-ci-success"]["needs"])
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
        "stage-application-wheel": "skipped",
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
        "stage-application-wheel": "skipped",
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
        "stage-application-wheel": "skipped",
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
    assert rust_plan["components"]["collector_data"] is False
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
        observed_components.update(actual)

    assert observed_components == set(COMPONENTS)


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


def test_prediction_gate_accepts_the_callers_event_name() -> None:
    refs = _workflow("prediction-regression-gate.yml")["jobs"]["refs"]
    commands = "\n".join(step.get("with", {}).get("script", "") for step in refs["steps"])

    assert 'process.env.OLD_REF_INPUT || "main"' in commands
    assert "unsupported event" not in commands


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


def test_collector_comparison_fetch_preserves_full_history() -> None:
    commands = _run_commands(_workflow("collector-check.yml")["jobs"]["check"])

    assert 'git fetch --no-tags origin "${BASE_SHA}"' in commands
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
        step.get("uses") == action_path for step in _workflow("fpe-support-matrix.yml")["jobs"]["generate"]["steps"]
    )


def test_macos_wheel_environment_seeds_pip_for_shared_verification() -> None:
    with (ACTION_ROOT / "build-platform-wheel" / "action.yml").open(encoding="utf-8") as handle:
        action = yaml.load(handle, Loader=yaml.BaseLoader)

    commands = _run_commands(action["runs"])
    assert 'uv venv --seed --python 3.13 "${RUNNER_TEMP}/aisimulate-wheel-venv"' in commands
    assert "python -m pip install --quiet wheelhouse/aisimulate-*.whl" in commands


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
