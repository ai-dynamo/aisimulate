# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

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
        "build-platform-wheels-copied-pr.yml",
        "collector-check.yml",
        "prediction-regression-gate.yml",
        "validate-platform-wheels.yml",
    }

    assert expected.issubset({path.name for path in WORKFLOW_ROOT.glob("*.yml")})


def test_full_ci_selects_migrated_contract_and_regression_suites() -> None:
    jobs = _workflow("ci.yml")["jobs"]

    assert {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
        "ci-success",
    }.issubset(jobs)

    application_commands = _run_commands(jobs["application-tests"])
    compatibility_commands = _run_commands(jobs["python-compatibility"])
    assert "python/aisimulate/tests/cross_package" in application_commands
    assert "python/aisimulate/tests/cross_package" in compatibility_commands
    assert "test_core_public_api.py" not in application_commands
    assert "test_core_public_api.py" not in compatibility_commands

    regression = jobs["engine-golden-regression"]
    regression_commands = _run_commands(regression)
    assert regression["name"] == "Engine golden regression"
    assert "test_engine_step_parity.py" in regression_commands
    assert "test_compile_engine_parity.py" in regression_commands
    assert regression_commands.count("-c python/aisimulate/pytest.ini") == 2

    feature_mode_commands = _run_commands(jobs["rust-feature-modes"])
    assert "cargo test --workspace --features embed-python,replay-bench" in feature_mode_commands
    assert "--all-features" not in feature_mode_commands
    assert "--no-default-features" not in feature_mode_commands
    assert "PYTHONPATH" not in feature_mode_commands

    migrated_gates = {
        "rust-feature-modes",
        "python-compatibility",
        "engine-golden-regression",
    }
    assert migrated_gates.issubset(set(jobs["ci-success"]["needs"]))
    assert migrated_gates.issubset(set(jobs["application-wheel"]["needs"]))


def test_full_ci_aggregate_checks_every_declared_dependency() -> None:
    aggregate = _workflow("ci.yml")["jobs"]["ci-success"]
    commands = _run_commands(aggregate)

    assert "stage-application-wheel" in aggregate["needs"]
    assert aggregate["steps"][0]["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert 'expected = {name: "success" for name in needs}' in commands
    assert 'expected["stage-application-wheel"] = os.environ["EXPECTED_STAGE_RESULT"]' in commands
    assert 'payload["result"]' in commands
    assert "Full CI did not pass" in commands


def test_privileged_pr_workflows_use_copied_pr_pushes() -> None:
    for filename in (
        "build-platform-wheels-copied-pr.yml",
        "collector-check.yml",
        "prediction-regression-gate.yml",
    ):
        triggers = _workflow(filename)["on"]
        assert "push" in triggers
        assert "pull_request" not in triggers


def test_manual_comparison_runs_have_unique_concurrency_groups() -> None:
    for filename in ("collector-check.yml", "prediction-regression-gate.yml"):
        group = _workflow(filename)["concurrency"]["group"]
        assert "github.event_name == 'workflow_dispatch'" in group
        assert "github.run_id" in group


def test_copied_pr_wheel_filter_covers_every_build_input() -> None:
    paths = set(_workflow("build-platform-wheels-copied-pr.yml")["on"]["push"]["paths"])

    assert {
        ".dockerignore",
        "THIRD_PARTY_NOTICES.md",
        "python/aisimulate/README.md",
        "python/aisimulate/THIRD_PARTY_NOTICES.md",
        "python/aisimulate/collector/**",
    }.issubset(paths)


def test_platform_wheel_build_and_verifiers_cover_collector_payload() -> None:
    dockerfile = (REPOSITORY_ROOT / "python" / "aisimulate" / "docker" / "Dockerfile").read_text()
    release_verifier = (REPOSITORY_ROOT / "python" / "aisimulate" / "tools" / "verify_release_wheels.py").read_text()
    installed_verifier = (
        REPOSITORY_ROOT / "python" / "aisimulate" / "tools" / "verify_installed_package_layers.py"
    ).read_text()

    assert "COPY python/aisimulate/collector/ /workspace/python/aisimulate/collector/" in dockerfile
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
    legacy_perf_files = tuple(
        (REPOSITORY_ROOT / "python" / "aisimulate" / "src" / "aiconfigurator_core" / "systems").rglob("*.txt")
    )

    assert generate["container"]["image"] == "${{ vars.CI_JOB_CONTAINER_IMAGE }}"
    assert checkout.get("with", {}).get("lfs") != "true", (
        "The current FPE job container does not provide git-lfs; update the CI image before enabling LFS checkout."
    )
    assert not legacy_perf_files, (
        "The FPE job container intentionally omits git-lfs. Before adding a legacy systems/**/*.txt asset, "
        "materialize LFS in CI or remove the stale .gitattributes rule."
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
