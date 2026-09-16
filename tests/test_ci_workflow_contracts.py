# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep migrated CI active at the repository root."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from scripts import build_manylinux_wheel as manylinux_builder
from scripts.build_manylinux_wheel import manylinux_platform
from scripts.check_application_test_inventory import Inventory, assignment
from scripts.require_fast_ci import REQUIRED_JOBS, GateError, latest_run, require_fast_ci, verify_jobs
from scripts.select_full_ci import COMPONENTS, select_components

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_ROOT = REPOSITORY_ROOT / ".github" / "actions"


def _fast_run(**overrides):
    return {
        "id": 10,
        "path": ".github/workflows/fast-ci.yml",
        "head_sha": "a" * 40,
        "head_branch": "main",
        "head_repository": {"full_name": "ai-dynamo/aisimulate"},
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "html_url": "https://github.com/ai-dynamo/aisimulate/actions/runs/10",
        **overrides,
    }


def _fast_jobs():
    return [
        {"name": name, "head_sha": "a" * 40, "status": "completed", "conclusion": "success"}
        for name in sorted(REQUIRED_JOBS)
    ]


def _require_fast(api, **kwargs):
    return require_fast_ci("ai-dynamo/aisimulate", "a" * 40, "refs/heads/main", "push", api=api, **kwargs)


def test_fast_prerequisite_accepts_all_jobs_from_the_latest_attempt_across_pages():
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        if "/jobs?" in endpoint:
            assert "/attempts/2/" in endpoint
            return [{"jobs": _fast_jobs()[:2]}, {"jobs": _fast_jobs()[2:]}]
        return [
            {"workflow_runs": [_fast_run(id=9, conclusion="failure")]},
            {"workflow_runs": [_fast_run(run_attempt=2)]},
        ]

    assert _require_fast(api)["run_attempt"] == 2
    assert len(calls) == 3


@pytest.mark.parametrize(
    "override",
    [
        {"head_sha": "b" * 40},
        {"head_branch": "release/old"},
        {"head_repository": {"full_name": "someone/aisimulate"}},
        {"path": ".github/workflows/ci.yml"},
        {"event": "pull_request"},
        {"event": "workflow_dispatch"},
    ],
)
def test_fast_prerequisite_rejects_wrong_commit_branch_origin_workflow_and_event(override):
    with pytest.raises(GateError, match="No complete Fast CI evidence"):
        _require_fast(lambda _: [{"workflow_runs": [_fast_run(**override)]}], timeout=0)


def test_manual_full_ci_can_reuse_a_same_branch_push_or_manual_fast_run():
    for event in ("push", "workflow_dispatch"):
        run = _fast_run(event=event)
        assert (
            latest_run([{"workflow_runs": [run]}], "a" * 40, "main", "workflow_dispatch", "ai-dynamo/aisimulate") == run
        )


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", "neutral", "timed_out", None])
def test_fast_prerequisite_cannot_fall_back_to_an_older_success(conclusion):
    pages = [{"workflow_runs": [_fast_run(), _fast_run(id=11, conclusion=conclusion)]}]
    with pytest.raises(GateError, match="Latest Fast CI run 11"):
        _require_fast(lambda _: pages)


@pytest.mark.parametrize(
    "override",
    [{"head_sha": "b" * 40}, {"status": "queued"}, {"conclusion": "skipped"}, {"conclusion": "failure"}],
)
def test_fast_prerequisite_rejects_partial_or_wrong_commit_job_evidence(override):
    jobs = _fast_jobs()
    jobs[0].update(override)
    with pytest.raises(GateError, match="wrong-commit job"):
        verify_jobs([{"jobs": jobs}], "a" * 40)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "aggregate-only"])
def test_fast_prerequisite_requires_substantive_jobs_not_just_a_green_aggregate(kind):
    jobs = _fast_jobs()
    jobs = jobs[1:] if kind == "missing" else jobs + [jobs[0]] if kind == "duplicate" else [jobs[0]]
    with pytest.raises(GateError, match="every required job"):
        verify_jobs([{"jobs": jobs}], "a" * 40)


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting"])
def test_fast_prerequisite_waits_with_a_bounded_timeout(status):
    now = [0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(GateError, match="within 20s"):
        _require_fast(
            lambda _: [{"workflow_runs": [_fast_run(status=status, conclusion=None)]}],
            timeout=20,
            interval=15,
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert now[0] == 20


def test_fast_prerequisite_rechecks_a_rerun_started_during_job_inspection():
    reads = [0]

    def api(endpoint):
        if "/jobs?" in endpoint:
            return [{"jobs": _fast_jobs()}]
        reads[0] += 1
        run = _fast_run() if reads[0] == 1 else _fast_run(run_attempt=2, conclusion="failure")
        return [{"workflow_runs": [run]}]

    with pytest.raises(GateError, match="Latest Fast CI run 10"):
        _require_fast(api, sleep=lambda _: None)


def test_fast_prerequisite_api_failure_never_becomes_success():
    def api(_):
        raise GateError("API unavailable")

    with pytest.raises(GateError, match="API unavailable"):
        _require_fast(api)


def _run_fast_prerequisite_cli(tmp_path, *, overrides=False, failure=None):
    run = _fast_run(head_branch="codex/manual", event="workflow_dispatch")
    jobs = _fast_jobs()
    if failure == "wrong-job-sha":
        jobs[0]["head_sha"] = "b" * 40
    fixture = tmp_path / "api.json"
    fixture.write_text(json.dumps({"run": run, "jobs": jobs}))
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "assert sys.argv[1:4] == ['api', '--paginate', '--slurp']\n"
        "if os.environ['TEST_API_FAILURE'] == 'true': sys.exit(1)\n"
        "data = json.loads(pathlib.Path(os.environ['TEST_API_FIXTURE']).read_text())\n"
        "with open(os.environ['TEST_API_CALLS'], 'a') as stream: stream.write(sys.argv[-1] + '\\n')\n"
        "print(json.dumps([{'jobs': data['jobs']} if '/jobs?' in sys.argv[-1] "
        "else {'workflow_runs': [data['run']]}]))\n"
    )
    gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    summary.write_text("Existing evidence\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "ai-dynamo/aisimulate",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_REF": "refs/heads/codex/manual",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_STEP_SUMMARY": str(summary),
        "TEST_API_FIXTURE": str(fixture),
        "TEST_API_CALLS": str(tmp_path / "calls"),
        "TEST_API_FAILURE": str(failure == "api").lower(),
    }
    args = [sys.executable, str(REPOSITORY_ROOT / "scripts/require_fast_ci.py"), "--timeout", "0"]
    if overrides:
        for option, variable in (
            ("--repository", "GITHUB_REPOSITORY"),
            ("--sha", "GITHUB_SHA"),
            ("--ref", "GITHUB_REF"),
            ("--event", "GITHUB_EVENT_NAME"),
        ):
            args.extend([option, env[variable]])
            env[variable] = "invalid-environment-default"
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False, timeout=10)
    return result, summary


@pytest.mark.parametrize("overrides", [False, True], ids=["environment-defaults", "cli-overrides"])
def test_fast_prerequisite_cli_verifies_manual_run_and_appends_summary(tmp_path, overrides):
    result, summary = _run_fast_prerequisite_cli(tmp_path, overrides=overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary.read_text().startswith("Existing evidence\n### Standalone Fast CI verified")
    for evidence in ("a" * 40, "https://github.com/ai-dynamo/aisimulate/actions/runs/10", "attempt 1"):
        assert evidence in result.stdout
        assert evidence in summary.read_text()
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[0] == calls[2]
    assert calls[0].startswith("repos/ai-dynamo/aisimulate/actions/workflows/fast-ci.yml/runs?")
    assert calls[1] == "repos/ai-dynamo/aisimulate/actions/runs/10/attempts/1/jobs?per_page=100"


@pytest.mark.parametrize("failure", ["api", "wrong-job-sha"])
def test_fast_prerequisite_cli_errors_leave_success_summary_unwritten(tmp_path, failure):
    result, summary = _run_fast_prerequisite_cli(tmp_path, failure=failure)
    assert result.returncode != 0
    assert "::error::" in result.stderr
    assert "Standalone Fast CI verified" not in result.stdout
    assert summary.read_text() == "Existing evidence\n"


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

    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
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


def test_linux_release_wheels_are_repaired_for_manylinux_2_28() -> None:
    expected_images = {
        "amd64": "quay.io/pypa/manylinux_2_28_x86_64@sha256:",
        "arm64": "quay.io/pypa/manylinux_2_28_aarch64@sha256:",
    }
    full_ci = _workflow("ci.yml")["jobs"]
    for job_name in ("application-test-wheel", "release-artifact-contract"):
        job = full_ci[job_name]
        assert job["container"]["image"] == "${{ matrix.container_image }}"
        images = {entry["arch"]: entry["container_image"] for entry in job["strategy"]["matrix"]["include"]}
        assert images.keys() == expected_images.keys()
        for arch, prefix in expected_images.items():
            assert images[arch].startswith(prefix)
            assert len(images[arch].removeprefix(prefix)) == 64

    application_commands = _run_commands(full_ci["application-test-wheel"])
    assert "scripts/build_manylinux_wheel.py" in application_commands
    assert "maturin build" not in application_commands

    fpe_prepare = _workflow("fpe-support-matrix.yml")["jobs"]["prepare-wheel"]
    assert fpe_prepare["container"]["image"].startswith(expected_images["amd64"])
    assert "scripts/build_manylinux_wheel.py" in _run_commands(fpe_prepare)

    release_builder = (REPOSITORY_ROOT / "scripts" / "build_release_artifacts.py").read_text()
    assert 'sys.platform.startswith("linux")' in release_builder
    assert '"build_manylinux_wheel.py"' in release_builder

    dockerfile = (REPOSITORY_ROOT / "python" / "aisimulate" / "docker" / "Dockerfile").read_text()
    assert 'MATURIN_PEP517_ARGS="--auditwheel skip"' in dockerfile
    assert "auditwheel repair" in dockerfile
    assert "auditwheel show /workspace/dist/aisimulate-*.whl" in dockerfile
    assert "--compatibility manylinux_2_28" not in dockerfile


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "manylinux_2_28_x86_64"),
        ("amd64", "manylinux_2_28_x86_64"),
        ("aarch64", "manylinux_2_28_aarch64"),
        ("arm64", "manylinux_2_28_aarch64"),
    ],
)
def test_manylinux_platform_maps_native_architectures(machine: str, expected: str) -> None:
    assert manylinux_platform(machine) == expected


def test_manylinux_platform_rejects_unknown_architecture() -> None:
    with pytest.raises(SystemExit, match="unsupported wheel architecture"):
        manylinux_platform("riscv64")


def _manylinux_build_case(
    tmp_path, monkeypatch, *, machine="x86_64", raw_count=1, repaired_count=1, repaired_tag=None, fail_repair=False
):
    monkeypatch.setattr(manylinux_builder.sys, "platform", "linux")
    monkeypatch.setattr(manylinux_builder.platform, "machine", lambda: machine)
    monkeypatch.setattr(manylinux_builder.shutil, "which", lambda _: "/usr/bin/auditwheel")
    calls = []
    output = tmp_path / "dist"

    def run(*command, cwd=REPOSITORY_ROOT):
        calls.append((command, cwd))
        if command[:3] == (sys.executable, "-m", "maturin"):
            raw_output = Path(command[command.index("--out") + 1])
            for index in range(raw_count):
                (raw_output / f"aisimulate-0.12.{index}-cp311-abi3-linux_{machine}.whl").touch()
        elif command[:2] == ("auditwheel", "repair"):
            if fail_repair:
                raise subprocess.CalledProcessError(1, command)
            tag = repaired_tag or f"manylinux_2_28_{machine}"
            for index in range(repaired_count):
                (output / f"aisimulate-0.12.{index}-cp311-abi3-{tag}.whl").touch()
        else:
            assert command[:2] == ("auditwheel", "show")

    monkeypatch.setattr(manylinux_builder, "_run", run)
    return output, calls


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_manylinux_build_repairs_the_exact_raw_wheel_then_audits_it(tmp_path, monkeypatch, machine):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, machine=machine)
    repaired = manylinux_builder.build(output)
    raw_output = calls[0][0][-1]
    raw_wheel = str(Path(raw_output) / f"aisimulate-0.12.0-cp311-abi3-linux_{machine}.whl")
    assert repaired == output / f"aisimulate-0.12.0-cp311-abi3-manylinux_2_28_{machine}.whl"
    assert list(output.iterdir()) == [repaired]
    assert calls == [
        (
            (sys.executable, "-m", "maturin", "build", "--release", "--auditwheel", "skip", "--out", raw_output),
            manylinux_builder.PYTHON_PROJECT,
        ),
        (
            ("auditwheel", "repair", "--plat", f"manylinux_2_28_{machine}", "--wheel-dir", str(output), raw_wheel),
            REPOSITORY_ROOT,
        ),
        (("auditwheel", "show", str(repaired)), REPOSITORY_ROOT),
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"raw_count": 0}, "exactly one unrepaired"),
        ({"raw_count": 2}, "exactly one unrepaired"),
        ({"repaired_count": 0}, "exactly one repaired"),
        ({"repaired_count": 2}, "exactly one repaired"),
        ({"repaired_tag": "manylinux_2_17_x86_64"}, "required tag manylinux_2_28_x86_64"),
    ],
)
def test_manylinux_build_rejects_missing_duplicate_or_wrong_policy_wheels(tmp_path, monkeypatch, options, message):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, **options)
    with pytest.raises(SystemExit, match=message):
        manylinux_builder.build(output)
    assert not any(command[:2] == ("auditwheel", "show") for command, _ in calls)


def test_manylinux_build_propagates_repair_failure(tmp_path, monkeypatch):
    output, calls = _manylinux_build_case(tmp_path, monkeypatch, fail_repair=True)
    with pytest.raises(subprocess.CalledProcessError):
        manylinux_builder.build(output)
    assert not any(command[:2] == ("auditwheel", "show") for command, _ in calls)


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


def test_fast_ci_is_standalone_with_an_exact_commit_prerequisite() -> None:
    fast_ci = _workflow("fast-ci.yml")
    assert "workflow_call" not in fast_ci["on"]
    assert set(fast_ci["on"]["push"]["branches"]) == {"main", "pull-request/*", "release/*"}
    assert fast_ci["on"]["workflow_dispatch"]["inputs"]["expected_sha"]["required"] == "true"
    gate = _workflow("ci.yml")["jobs"]["fast-ci"]
    assert gate["name"] == "Require Fast CI"
    assert "uses" not in gate
    assert gate["permissions"] == {"contents": "read", "actions": "read"}
    assert "scripts/require_fast_ci.py" in _run_commands(gate)
    assert "workflow_run" not in _workflow("ci.yml")["on"]
    whitespace = _run_commands(fast_ci["jobs"]["python-static"])
    assert "[.head.sha, .base.sha] | @tsv" in whitespace
    assert fast_ci["jobs"]["python-static"]["env"]["BASE_SHA"] == (
        "${{ inputs.base_sha || github.event.pull_request.base.sha || github.event.before }}"
    )


def test_fast_ci_missing_base_fetch_uses_temporary_checkout_authentication() -> None:
    workflow = _workflow("fast-ci.yml")
    job = workflow["jobs"]["python-static"]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    whitespace = next(step for step in job["steps"] if step.get("name") == "Check changed-line whitespace")

    assert workflow["permissions"]["contents"] == "read"
    assert checkout["with"]["persist-credentials"] == "false"
    assert whitespace["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "printf 'x-access-token:%s' \"${GH_TOKEN}\"" in whitespace["run"]
    assert (
        'git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${basic_token}" fetch' in whitespace["run"]
    )
    assert '--no-tags origin "${BASE_SHA}"' in whitespace["run"]


def _run_pr_target(
    tmp_path: Path, head: str, base: str, api_status: int = 0
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    step = next(step for step in _workflow("ci.yml")["jobs"]["verify-target"]["steps"] if step.get("id") == "pr-target")
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "${GH_CALLED}"\n'
        'if [[ "${TEST_API_STATUS}" != 0 ]]; then exit "${TEST_API_STATUS}"; fi\n'
        'printf "%s\\t%s\\n" "${TEST_PR_HEAD}" "${TEST_PR_BASE}"\n'
    )
    gh.chmod(0o755)
    output_path = tmp_path / "pr-output"
    calls_path = tmp_path / "gh-called"
    result = _run_workflow_script(
        step["run"],
        {
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/pull-request/121",
            "RUN_SHA": "a" * 40,
            "REPOSITORY": "ai-dynamo/aisimulate",
            "GITHUB_OUTPUT": str(output_path),
            "GH_CALLED": str(calls_path),
            "TEST_PR_HEAD": head,
            "TEST_PR_BASE": base,
            "TEST_API_STATUS": str(api_status),
        },
    )
    assert calls_path.read_text().splitlines() == [
        "api -X GET repos/ai-dynamo/aisimulate/pulls/121 --jq [.head.sha, .base.sha] | @tsv"
    ]
    output = dict(line.split("=", 1) for line in output_path.read_text().splitlines()) if output_path.exists() else {}
    return result, output


def test_full_ci_trusted_copy_validates_the_stacked_pr_base(tmp_path: Path) -> None:
    result, output = _run_pr_target(tmp_path, "a" * 40, "d" * 40)
    assert result.returncode == 0, result.stderr
    assert output == {}


@pytest.mark.parametrize(
    ("head", "base", "api_status"),
    [
        pytest.param("b" * 40, "d" * 40, 0, id="head-changed"),
        pytest.param("a" * 40, "main", 0, id="mutable-base"),
        pytest.param("a" * 40, "d" * 39, 0, id="short-base"),
        pytest.param("a" * 40, "", 0, id="missing-base"),
        pytest.param("a" * 40, "d" * 40, 1, id="api-failure"),
    ],
)
def test_full_ci_trusted_copy_base_fails_closed(tmp_path: Path, head: str, base: str, api_status: int) -> None:
    result, output = _run_pr_target(tmp_path, head, base, api_status)
    assert result.returncode != 0
    assert output == {}


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI contract test",
            "-c",
            "user.email=ci-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        capture_output=True,
        text=True,
        check=check,
    )


def _commit_file(repository: Path, name: str, contents: str) -> str:
    (repository / name).write_text(contents)
    _git(repository, "add", name)
    _git(repository, "commit", "--quiet", "-m", name)
    return _git(repository, "rev-parse", "HEAD").stdout.strip()


def _run_whitespace_step(
    repository: Path,
    base: str,
    target: str,
    ref: str = "refs/heads/pull-request/121",
    event: str = "push",
) -> subprocess.CompletedProcess[str]:
    job = _workflow("fast-ci.yml")["jobs"]["python-static"]
    script = next(step["run"] for step in job["steps"] if step.get("name") == "Check changed-line whitespace")
    gh = repository / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\t%s\\n" "$TARGET_SHA" "$TEST_PR_BASE"\n')
    gh.chmod(0o755)
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=repository,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "BEFORE_SHA": "f" * 40,
            "BASE_SHA": base,
            "TARGET_SHA": target,
            "GH_TOKEN": "ci-contract-dummy-token",
            "PATH": f"{repository}{os.pathsep}{os.environ['PATH']}",
            "REPOSITORY": "ai-dynamo/aisimulate",
            "TEST_PR_BASE": base,
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("case", ["matching", "missing-expected", "wrong-run-sha", "wrong-checkout"])
def test_fast_ci_manual_exact_target_guard(tmp_path: Path, case: str) -> None:
    _git(tmp_path, "init", "--quiet")
    head = _commit_file(tmp_path, "root.txt", "root\n")
    job = _workflow("fast-ci.yml")["jobs"]["policy"]
    assert job["env"]["EXPECTED_SHA"] == "${{ inputs.expected_sha }}"
    assert job["env"]["TARGET_SHA"] == "${{ inputs.expected_sha || github.event.pull_request.head.sha || github.sha }}"
    step = next(step for step in job["steps"] if step.get("name") == "Verify exact target")
    expected = "" if case == "missing-expected" else "b" * 40 if case == "wrong-checkout" else head
    target = expected or head
    run_sha = "b" * 40 if case in {"wrong-run-sha", "wrong-checkout"} else head
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "EXPECTED_SHA": expected,
            "TARGET_SHA": target,
            "GITHUB_SHA": run_sha,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (case == "matching"), result.stdout + result.stderr


@pytest.mark.parametrize("supplied_base", [False, True], ids=["default-last-commit", "explicit-whole-change"])
def test_fast_ci_manual_base_selects_the_whitespace_range(tmp_path: Path, supplied_base: bool) -> None:
    _git(tmp_path, "init", "--quiet")
    base = _commit_file(tmp_path, "root.txt", "root\n")
    _commit_file(tmp_path, "first.txt", "earlier whitespace error \n")
    target = _commit_file(tmp_path, "last.txt", "clean final commit\n")
    workflow = _workflow("fast-ci.yml")
    base_input = workflow["on"]["workflow_dispatch"]["inputs"]["base_sha"]
    assert base_input["required"] == "false"
    assert base_input["type"] == "string"
    assert base_input["default"] == ""
    # A manual event has neither pull_request.base.sha nor before; the workflow
    # expression selects this input or its empty default.
    assert workflow["jobs"]["python-static"]["env"]["BASE_SHA"] == (
        "${{ inputs.base_sha || github.event.pull_request.base.sha || github.event.before }}"
    )
    result = _run_whitespace_step(
        tmp_path,
        base if supplied_base else base_input["default"],
        target,
        ref="refs/heads/codex/manual",
        event="workflow_dispatch",
    )
    assert (result.returncode != 0) is supplied_base, result.stdout + result.stderr
    assert ("first.txt" in result.stdout) is supplied_base


@pytest.mark.parametrize("earlier_whitespace", [False, True], ids=["clean-rebased-copy", "earlier-pr-commit"])
def test_fast_ci_whitespace_checks_the_whole_pr_above_its_stacked_base(
    tmp_path: Path, earlier_whitespace: bool
) -> None:
    _git(tmp_path, "init", "--quiet")
    _commit_file(tmp_path, "root.txt", "root\n")
    base = _commit_file(tmp_path, "parent-pr.txt", "outside this PR \n")
    _commit_file(tmp_path, "first-pr.txt", "first change \n" if earlier_whitespace else "first change\n")
    target = _commit_file(tmp_path, "last-pr.txt", "last change\n")
    assert _git(tmp_path, "cat-file", "-e", f"{'f' * 40}^{{commit}}", check=False).returncode != 0
    assert _git(tmp_path, "diff", "--check", f"{target}^", target).returncode == 0

    result = _run_whitespace_step(tmp_path, base, target)

    assert (result.returncode != 0) is earlier_whitespace, result.stdout + result.stderr
    assert "parent-pr.txt" not in result.stdout
    if earlier_whitespace:
        assert "first-pr.txt" in result.stdout
        assert "trailing whitespace" in result.stdout


@pytest.mark.parametrize("base", ["", "0" * 40], ids=["manual-run", "new-release-branch"])
def test_fast_ci_without_a_base_preserves_the_last_commit_range(tmp_path: Path, base: str) -> None:
    _git(tmp_path, "init", "--quiet")
    _commit_file(tmp_path, "earlier.txt", "outside the last commit \n")
    target = _commit_file(tmp_path, "last.txt", "last change\n")

    result = _run_whitespace_step(tmp_path, base, target, ref="refs/heads/release/new")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "earlier.txt" not in result.stdout


@pytest.mark.parametrize("available", [True, False], ids=["fetch-base-history", "missing-base-fails-closed"])
def test_fast_ci_whitespace_fetches_an_absent_base_without_truncating_history(tmp_path: Path, available: bool) -> None:
    origin = tmp_path / "origin"
    checkout = tmp_path / "checkout"
    origin.mkdir()
    checkout.mkdir()
    _git(origin, "init", "--quiet")
    common = _commit_file(origin, "root.txt", "root\n")
    target = _commit_file(origin, "pr.txt", "PR change\n")
    _git(origin, "checkout", "--quiet", "-b", "base", common)
    _commit_file(origin, "base-first.txt", "first base change\n")
    base = _commit_file(origin, "base-last.txt", "last base change\n")
    _git(origin, "tag", "not-needed-for-whitespace", base)
    _git(checkout, "init", "--quiet")
    _git(checkout, "remote", "add", "origin", origin.as_uri())
    _git(checkout, "fetch", "--no-tags", "origin", target)
    _git(checkout, "checkout", "--quiet", "--detach", target)
    assert _git(checkout, "cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0

    result = _run_whitespace_step(checkout, base if available else "f" * 40, target)

    if available:
        assert result.returncode == 0, result.stdout + result.stderr
        assert _git(checkout, "merge-base", base, target).stdout.strip() == common
        assert _git(checkout, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
        assert not _git(checkout, "tag", "--list").stdout.strip()
        assert _git(checkout, "config", "--local", "--get-regexp", "extraheader", check=False).returncode == 1
    else:
        assert result.returncode != 0
        assert "upload-pack: not our ref" in result.stderr
        assert "Invalid symmetric difference expression" not in result.stderr


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
        "python-compatibility",
        "engine-golden-regression",
    ):
        assert any(step.get("uses") == action_path for step in full_ci[job_name]["steps"])

    assert any(step.get("uses") == action_path for step in _workflow("collector-check.yml")["jobs"]["check"]["steps"])


def test_manylinux_jobs_select_bundled_python_without_setup_python() -> None:
    action_path = "./.github/actions/setup-manylinux-python-rust"
    for workflow_name, job_name in (
        ("ci.yml", "application-test-wheel"),
        ("ci.yml", "release-artifact-contract"),
        ("fpe-support-matrix.yml", "prepare-wheel"),
    ):
        steps = _workflow(workflow_name)["jobs"][job_name]["steps"]
        assert any(step.get("uses") == action_path for step in steps)
        assert not any(
            step.get("uses", "").startswith(("actions/setup-python@", "./.github/actions/setup-python-rust"))
            for step in steps
        )
    action = yaml.safe_load((ACTION_ROOT / "setup-manylinux-python-rust" / "action.yml").read_text())
    assert action["runs"]["steps"][0]["env"]["MANYLINUX_PYTHON_ROOT"] == "/opt/python/cp312-cp312"
    assert not any(step.get("uses", "").startswith("actions/setup-python@") for step in action["runs"]["steps"])
    rust = next(step for step in action["runs"]["steps"] if step.get("uses", "").startswith("dtolnay/rust-toolchain@"))
    assert rust["with"]["toolchain"] == _workflow("nightly-ci.yml")["jobs"]["build-artifacts"]["env"]["RUST_TOOLCHAIN"]


@pytest.mark.parametrize(
    "missing", [None, "python", "cc", "c++", "make", "auditwheel", "patchelf", "auditwheel-broken", "patchelf-broken"]
)
def test_manylinux_bootstrap_exports_working_python_or_fails_before_build(tmp_path: Path, missing: str | None) -> None:
    action = yaml.safe_load((ACTION_ROOT / "setup-manylinux-python-rust" / "action.yml").read_text())
    bootstrap = action["runs"]["steps"][0]["run"]
    python_root = tmp_path / "bundled-python"
    binaries = python_root / "bin"
    binaries.mkdir(parents=True)
    if missing != "python":
        (binaries / "python").symlink_to(sys.executable)
    tools = tmp_path / "tools"
    tools.mkdir()
    tool_log = tmp_path / "tool-log"
    for name in ("cc", "c++", "make", "auditwheel", "patchelf"):
        if name != missing:
            binary = tools / name
            if name in ("auditwheel", "patchelf"):
                binary.write_text(
                    f'#!/bin/bash\necho "{name} $*" >> "$AUDIT_TOOL_LOG"\n'
                    + ("exit 1\n" if missing == f"{name}-broken" else 'test "$1" = --version\n')
                )
                binary.chmod(0o755)
            else:
                binary.symlink_to("/usr/bin/true")
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    env_file.touch()
    path_file.touch()
    runner_temp = tmp_path / "runner-temp"
    result = subprocess.run(
        ["/bin/bash", "-c", bootstrap],
        env={
            **os.environ,
            "PATH": str(tools),
            "MANYLINUX_PYTHON_ROOT": str(python_root),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(env_file),
            "GITHUB_PATH": str(path_file),
            "AUDIT_TOOL_LOG": str(tool_log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if missing:
        assert result.returncode != 0
        assert env_file.read_text() == path_file.read_text() == ""
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert tool_log.read_text().splitlines() == ["auditwheel --version", "patchelf --version"]
    exported = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert exported == {
        "PYO3_PYTHON": str(binaries / "python"),
        "LIBRARY_PATH": str(python_root / "lib"),
        "CARGO_HOME": str(runner_temp / "cargo"),
        "RUSTUP_HOME": str(runner_temp / "rustup"),
    }
    assert path_file.read_text().splitlines() == [str(binaries)]
    subprocess.run(
        ["python", "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
        env={**os.environ, **exported, "PATH": path_file.read_text().strip()},
        check=True,
        timeout=10,
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
    aggregate = _workflow("ci.yml")["jobs"]["readiness"]
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
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
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
        "python-compliance": "skipped",
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
        "python-compliance": "skipped",
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
        "python-compliance": "skipped",
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
    }.issubset(jobs["readiness"]["needs"])
    assert set(jobs["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}
    for suite in ("unit", "cli-build"):
        steps = [
            step for step in jobs["application-tests"]["steps"] if step.get("if") == f"matrix.shard.suite == '{suite}'"
        ]
        assert len(steps) == 1
        command = steps[0]["run"]
        assert command.count("--splits ") == 1
        assert command.count("--group ") == 1
        assert command.count("--splitting-algorithm ") == 1
        assert "--splits ${{ matrix.shard.groups }}" in command
        assert "--group ${{ matrix.shard.group }}" in command
        assert "--splitting-algorithm least_duration" in command


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


def test_pages_release_completion_still_executes_main_checkout():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/pages.yml").read_text())
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["workflow_run"]["branches"] == ["main", "release/**"]
    checkout = workflow["jobs"]["build"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event_name == 'pull_request' && github.ref || 'main' }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    watched_workflows = trigger["workflow_run"]["workflows"]
    for producer in ("fpe-support-matrix.yml", "nightly-ci.yml", "release-nightly-ci.yml"):
        assert _workflow(producer)["name"] in watched_workflows
    assert "Nightly CI" in watched_workflows  # Runs started before the main workflow rename.


def test_release_nightly_discovers_versions_and_bounds_total_concurrency():
    workflow = _workflow("release-nightly-ci.yml")
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["schedule"] == [{"cron": "23 9 * * *"}]
    assert "workflow_dispatch" in trigger
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {"group": "release-nightly-ci", "cancel-in-progress": "false"}
    discover = workflow["jobs"]["discover"]
    assert discover["if"] == "github.ref == 'refs/heads/main'"
    checkout = discover["steps"][0]["with"]
    assert checkout == {"ref": "${{ github.sha }}", "fetch-depth": "0", "persist-credentials": "false"}
    assert "run_release_fpe.py list-releases" in _run_commands(discover)
    releases = workflow["jobs"]["releases"]
    assert releases["needs"] == "discover"
    assert releases["if"] == "needs.discover.outputs.releases != '[]'"
    assert releases["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "1",
        "matrix": {"release": "${{ fromJSON(needs.discover.outputs.releases) }}"},
    }
    assert releases["uses"] == "./.github/workflows/fpe-release-qualify.yml"
    assert releases["with"] == {
        "release": "${{ matrix.release.version }}",
        "source_sha": "${{ matrix.release.source_sha }}",
    }


def test_release_qualification_pins_source_and_tooling_across_all_jobs():
    workflow = _workflow("fpe-release-qualify.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"] == {
        "FPE_BRANCH": "release/${{ inputs.release }}",
        "FPE_SOURCE_SHA": "${{ inputs.source_sha }}",
        "FPE_TOOLING_SHA": "${{ github.sha }}",
    }
    jobs = workflow["jobs"]
    assert jobs["prepare"]["if"] == "github.ref == 'refs/heads/main'"
    assert jobs["generate"]["needs"] == "prepare"
    assert set(jobs["qualify"]["needs"]) == {"prepare", "generate"}
    for job in jobs.values():
        checkouts = [step["with"] for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")]
        assert len(checkouts) == 2
        assert checkouts[0]["ref"] == "${{ github.sha }}"
        assert checkouts[1]["ref"] == "${{ inputs.source_sha }}"
        assert checkouts[1]["path"] == "release-source"
        assert all(step["persist-credentials"] == "false" for step in checkouts)
    prepare = _run_commands(jobs["prepare"])
    assert "maturin build --release --locked" in prepare
    assert "run_release_fpe.py record-wheel" in prepare
    assert "run_release_fpe.py discover" in prepare
    assert "uv sync --project release-source/python/aisimulate --locked --no-install-project --no-dev" in prepare
    generate = _run_commands(jobs["generate"])
    assert "run_release_fpe.py probe" in generate
    assert "--max-workers 8" in generate
    assert "--max-topologies-per-role" not in generate
    assert jobs["generate"]["strategy"]["max-parallel"] == "20"
    assert "run_release_fpe.py package" in _run_commands(jobs["qualify"])
    upload = jobs["qualify"]["steps"][-1]
    assert upload["with"]["name"] == "fpe-support-matrix-web-release-${{ inputs.release }}"
    assert upload["with"]["retention-days"] == "90"
    assert "if" not in upload  # Never publish a partially failed qualification.


def test_release_artifact_handoffs_cannot_mix_versions():
    import fnmatch

    jobs = _workflow("fpe-release-qualify.yml")["jobs"]
    wheel = jobs["prepare"]["steps"][-1]["with"]["name"]
    for name in ("generate", "qualify"):
        download = next(s for s in jobs[name]["steps"] if s.get("uses", "").startswith("actions/download-artifact@"))
        assert download["with"]["name"] == wheel
    assert wheel == "fpe-release-wheel-${{ inputs.release }}"
    upload = jobs["generate"]["steps"][-1]["with"]["name"]
    pattern = next(s["with"]["pattern"] for s in jobs["qualify"]["steps"] if "pattern" in s.get("with", {}))
    versions = ["0.12.0", "0.13.0", "0.13.0-rc1", "0.13.0--preview"]
    for selected in versions:
        selected_pattern = pattern.replace("${{ inputs.release }}", selected)
        for version in versions:
            name = (
                upload.replace("${{ inputs.release }}", version)
                .replace("${{ matrix.shard.system }}", "h200_sxm")
                .replace("${{ matrix.shard.backend }}", "vllm")
            )
            assert fnmatch.fnmatchcase(name, selected_pattern) == (version == selected)


def _nightly_condition(job: str, **overrides) -> bool:
    """Evaluate the checked-in predicate with explicit Actions context values."""
    values = {
        "github.event_name": "schedule",
        "github.ref": "refs/heads/main",
        "github.run_attempt": "1",
        "needs.manual-approval.result": "skipped",
        "needs.manual-approval.outputs.approved-attempt": "",
        "needs.changes-guard.outputs.should-build": "true",
        "vars.GITLAB_SECURITY_TRIGGER_ENABLED": "true",
        **{
            f"needs.{name}.result": "success"
            for name in ("build-artifacts", "python-compliance", "fpe-support-matrix", "license-evidence")
        },
        **overrides,
    }
    expression = _workflow("nightly-ci.yml")["jobs"][job]["if"]
    expression = re.sub(r"(?:github|needs|vars)\.[\w.-]+", lambda match: repr(values[match[0]]), expression)
    expression = expression.replace("!cancelled()", "True").replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}})


@pytest.mark.parametrize("job", ["build-artifacts", "trigger-gitlab-security"])
def test_nightly_retries_require_approval_from_the_current_attempt(job):
    assert _nightly_condition(job)
    for attempt in ("2", "3"):
        assert not _nightly_condition(job, **{"github.run_attempt": attempt})
        assert not _nightly_condition(
            job,
            **{
                "github.run_attempt": attempt,
                "needs.manual-approval.result": "success",
                "needs.manual-approval.outputs.approved-attempt": str(int(attempt) - 1),
            },
        )
        assert _nightly_condition(
            job,
            **{
                "github.run_attempt": attempt,
                "needs.manual-approval.result": "success",
                "needs.manual-approval.outputs.approved-attempt": attempt,
            },
        )
    push = {"github.event_name": "push", "github.ref": "refs/heads/pull-request/59"}
    assert not _nightly_condition(job, **push)
    assert _nightly_condition(
        job,
        **push,
        **{
            "needs.manual-approval.result": "success",
            "needs.manual-approval.outputs.approved-attempt": "1",
        },
    ) == (job == "build-artifacts")


@pytest.mark.parametrize("gate", ["build-artifacts", "fpe-support-matrix", "license-evidence"])
@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_nightly_failed_validation_cannot_publish(gate, result):
    assert not _nightly_condition("trigger-gitlab-security", **{f"needs.{gate}.result": result})


@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_nightly_failed_compliance_cannot_stage(result):
    assert not _nightly_condition("build-artifacts", **{"needs.python-compliance.result": result})


def test_nightly_dependency_execution_cannot_modify_staged_artifacts_or_inherit_deploy_secrets():
    jobs = _workflow("nightly-ci.yml")["jobs"]
    compliance = jobs["python-compliance"]
    assert "environment" not in compliance
    assert not re.search(r"\$\{\{\s*secrets\.", json.dumps(compliance))
    assert "pip-licenses" in _run_commands(compliance)
    assert "--with-system" in _run_commands(compliance)
    assert not any("download-artifact@" in s.get("uses", "") for s in compliance["steps"])
    assert all(
        s["with"]["path"].endswith("/*.csv") for s in compliance["steps"] if "upload-artifact@" in s.get("uses", "")
    )
    build = jobs["build-artifacts"]
    assert "python-compliance" in build["needs"]
    assert "pip-licenses" not in _run_commands(build)
    steps = build["steps"]
    stage_index = next(i for i, s in enumerate(steps) if s.get("name") == "Stage to Artifactory")
    fetch_index = next(
        i for i, s in enumerate(steps) if s.get("name") == "Fetch the staged wheel back from Artifactory"
    )
    smoke_index = next(
        i for i, s in enumerate(steps) if s.get("name") == "Smoke-test the staged wheel on every supported Python"
    )
    assert stage_index < fetch_index < smoke_index
    for step in steps[:stage_index]:
        if "pip install" in step.get("run", ""):
            assert "--require-hashes" in step["run"]
    assert not re.search(r"\$\{\{\s*secrets\.", json.dumps(steps[smoke_index:]))
    assert "ARTIFACTORY_TOKEN" not in steps[smoke_index].get("env", {})
    evidence = jobs["license-evidence"]
    assert "environment" not in evidence
    assert "pip install" not in _run_commands(evidence)
    assert set(evidence["needs"]) == {"build-artifacts", "python-compliance"}


def _nightly_license_report(tmp_path, crates, prior=None):
    inventories = tmp_path / "python"
    inventories.mkdir(exist_ok=True)
    # Same runtime dependency in multiple Python/architecture inventories must
    # be deduplicated, while concurrent versions of a crate must survive.
    for arch in ("amd64", "arm64"):
        (inventories / f"{arch}-3.12.csv").write_text("Name,Version,License\nprettytable,3.16.0,BSD-3-Clause\n")
    (tmp_path / "cargo-metadata.json").write_text(
        json.dumps(
            {
                "packages": [
                    {"id": f"getrandom@{version}", "name": "getrandom", "version": version, "license": spdx}
                    for version, spdx in crates
                ]
            }
        )
    )
    prior_csv = io.StringIO()
    writer = csv.DictWriter(prior_csv, fieldnames=["dependency_type", "name", "version", "spdx_license"])
    writer.writeheader()
    writer.writerows(prior or [])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("deps.csv", prior_csv.getvalue())

    def urlopen(request, timeout):
        assert timeout == 30
        if "/runs?" in request.full_url:
            payload = (
                {"workflow_runs": [{"id": 1, "artifacts_url": "https://fixture/artifacts"}]}
                if prior
                else {"workflow_runs": []}
            )
        elif request.full_url == "https://fixture/artifacts":
            payload = {"artifacts": [{"name": "license-artifacts", "archive_download_url": "https://fixture/archive"}]}
        elif request.full_url == "https://fixture/archive":
            return io.BytesIO(archive.getvalue())
        else:
            raise AssertionError(request.full_url)
        return io.BytesIO(json.dumps(payload).encode())

    step = next(
        s
        for s in _workflow("nightly-ci.yml")["jobs"]["license-evidence"]["steps"]
        if s.get("name") == "Generate license compliance evidence"
    )
    source = step["run"].split("<<'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
    with (
        patch.object(sys, "argv", ["nightly-evidence", str(tmp_path)]),
        patch.dict(os.environ, {"GH_API_TOKEN": "fixture", "GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": "2"}),
        patch("urllib.request.urlopen", side_effect=urlopen),
    ):
        exec(compile(source, "nightly-ci-evidence", "exec"), {})
    with (tmp_path / "deps.csv").open() as inventory, (tmp_path / "deps-diff.csv").open() as difference:
        return list(csv.DictReader(inventory)), list(csv.DictReader(difference))


def test_nightly_license_diff_preserves_concurrent_versions_and_license_changes(tmp_path):
    original = [("0.2.17", "MIT"), ("0.3.4", "MIT"), ("0.4.3", "MIT")]
    inventory, difference = _nightly_license_report(tmp_path, original)
    assert len(inventory) == len(difference) == 4
    assert {r["version"] for r in difference if r["dependency_type"] == "crate"} == {"0.2.17", "0.3.4", "0.4.3"}
    assert all(r["change"] == "added" for r in difference)
    assert _nightly_license_report(tmp_path, original, inventory)[1] == []
    removed = _nightly_license_report(tmp_path, original[1:], inventory)[1]
    assert len(removed) == 1
    assert (removed[0]["change"], removed[0]["prior_version"]) == ("removed", "0.2.17")
    changed = _nightly_license_report(tmp_path, [("0.2.17", "Apache-2.0"), *original[1:]], inventory)[1]
    assert len(changed) == 1
    assert (changed[0]["change"], changed[0]["version"], changed[0]["prior_spdx_license"]) == (
        "changed",
        "0.2.17",
        "MIT",
    )


@pytest.mark.parametrize(
    "conclusion,expected", [("failure", True), ("timed_out", True), ("success", False), ("skipped", False)]
)
def test_nightly_alert_collector_includes_timeouts(tmp_path, conclusion, expected):
    step = next(s for s in _workflow("nightly-ci.yml")["jobs"]["notify-slack"]["steps"] if s.get("id") == "failed")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text('#!/bin/sh\nprintf "%s\\n" "$FIXTURE_JOBS"\n')
    fake_curl.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GH_TOKEN": "fixture",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_RUN_ID": "1",
            "GITHUB_ENV": str(tmp_path / "env"),
            "GITHUB_OUTPUT": str(output),
            "MENTION_IDS": "",
            "FIXTURE_JOBS": json.dumps({"jobs": [{"name": "Build artifacts", "conclusion": conclusion}]}),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().strip() == f"has_failures={str(expected).lower()}"


def test_full_ci_license_failure_blocks_readiness_and_staging():
    jobs = _workflow("ci.yml")["jobs"]
    assert "python-compliance" in jobs["readiness"]["needs"]
    assert "readiness" in jobs["stage-application-wheel"]["needs"]
    plan = dict.fromkeys(COMPONENTS, "true")
    results = dict.fromkeys(jobs["readiness"]["needs"], "success")
    passed = _run_full_ci_aggregate(results, plan)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    results["python-compliance"] = "failure"
    failed = _run_full_ci_aggregate(results, plan)
    assert failed.returncode != 0
    assert "python-compliance=failure, expected success" in failed.stdout
