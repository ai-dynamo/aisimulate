# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-specific routing contract for AISimulate's generated CODEOWNERS."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from codeowners_match import parse_codeowners, resolve_owners

ROOT = Path(__file__).resolve().parents[2]
MAINTAINERS = "@ai-dynamo/access-aisimulate-maintain"
FPE = "@ai-dynamo/aisimulate-forward-pass-engine-codeowners"
SWEEPER = "@ai-dynamo/aisimulate-sweeper-codeowners"
REPLAY = "@ai-dynamo/aisimulate-replay-codeowners"
MOCKER = "@ai-dynamo/aisimulate-mocker-codeowners"
INFRA = "@ai-dynamo/aisimulate-infra-codeowners"
AREA_TEAMS = {FPE, SWEEPER, REPLAY, MOCKER}


def _owners(path: str) -> set[str]:
    rules = parse_codeowners((ROOT / "CODEOWNERS").read_text())
    return set(resolve_owners(rules, path))


def _run_readiness_script(script: str, tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    summary = tmp_path / "summary.md"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": "0123456789abcdef",
            "GITHUB_STEP_SUMMARY": str(summary),
            **env,
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_subsystem_teams_retain_maintainer_coownership() -> None:
    rules = parse_codeowners((ROOT / "CODEOWNERS").read_text())
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
    violations = []
    for path in tracked:
        owners = set(resolve_owners(rules, path))
        if owners & AREA_TEAMS and MAINTAINERS not in owners:
            violations.append(f"{path}: {sorted(owners)}")
    assert not violations


def test_representative_routing_contract() -> None:
    # Migrated AIC estimator and full application surface.
    assert _owners("crates/core/src/perfmodel/fpm/model.rs") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("crates/core/tests/perfmodel/memory_round_trip.rs") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aiconfigurator_core/sdk/engine.py") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aiconfigurator/generator/__init__.py") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/collector/collect.py") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("crates/tests/public-api/src/lib.rs") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("docs/core-api.md") == {FPE, MAINTAINERS}

    # Unified application Replay, Sweeper, and Mocker surface.
    assert _owners("python/aisimulate/src/aisimulate/aic.py") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aisimulate/sweeper/search.py") == {
        SWEEPER,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aisimulate/replay/cli.py") == {
        REPLAY,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aisimulate/runner.py") == {
        REPLAY,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aisimulate/traffic.py") == {
        REPLAY,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/src/aisimulate/__init__.py") == {
        FPE,
        SWEEPER,
        REPLAY,
        MAINTAINERS,
    }
    assert _owners("crates/core/src/replay/event.rs") == {REPLAY, MAINTAINERS}
    assert _owners("crates/core/src/engine/scheduler/vllm/core.rs") == {
        MOCKER,
        MAINTAINERS,
    }
    assert _owners("crates/core/src/engine/timing.rs") == {
        MOCKER,
        FPE,
        MAINTAINERS,
    }
    assert _owners("crates/core/src/python.rs") == {
        MOCKER,
        REPLAY,
        FPE,
        MAINTAINERS,
    }
    assert _owners("crates/core/Cargo.toml") == {
        MOCKER,
        REPLAY,
        FPE,
        INFRA,
        MAINTAINERS,
    }

    # Active and imported repository metadata.
    assert _owners(".github/workflows/ci.yml") == {INFRA}
    assert _owners(".github/workflows/fast-ci.yml") == {INFRA}
    assert _owners(".gitattributes") == {INFRA, MAINTAINERS}
    assert _owners("scripts/build_release_artifacts.py") == {INFRA, MAINTAINERS}
    assert _owners("tests/test_source_compliance.py") == {INFRA}
    assert _owners("python/aisimulate/.github/workflows/build-test.yml") == {
        INFRA,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/pyproject.toml") == {
        FPE,
        SWEEPER,
        REPLAY,
        MOCKER,
        INFRA,
        MAINTAINERS,
    }
    assert _owners(".github/workflows/codeowners.yml") == {INFRA, MAINTAINERS}
    assert _owners(".github/codeowners/areas.yaml") == {INFRA, MAINTAINERS}
    assert _owners("python/aisimulate/.github/codeowners/areas.yaml") == {
        INFRA,
        MAINTAINERS,
    }
    assert _owners("crates/core/deny.toml") == {
        FPE,
        MOCKER,
        REPLAY,
        INFRA,
        MAINTAINERS,
    }
    assert _owners("deny.toml") == {INFRA, MAINTAINERS}
    assert _owners("CODEOWNERS") == {INFRA, MAINTAINERS}
    assert _owners("AGENTS.md") == {INFRA, MAINTAINERS}
    assert _owners(".coderabbit.yaml") == {INFRA, MAINTAINERS}
    assert _owners("REVIEW.md") == {INFRA, MAINTAINERS}
    assert _owners("DEVELOPMENT.md") == {INFRA, MAINTAINERS}
    assert _owners("CODE_OF_CONDUCT.md") == {MAINTAINERS}
    assert _owners("README.md") == {MAINTAINERS}
    assert _owners("SECURITY.md") == {INFRA, MAINTAINERS}
    assert _owners("CONTRIBUTING.md") == {MAINTAINERS}


def test_unclassified_future_path_uses_maintainer_fallback() -> None:
    assert _owners("future/unclassified.txt") == {MAINTAINERS}


def test_dependency_policy_covers_every_rust_manifest_root() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for manifest in (
        "Cargo.toml",
        "crates/tests/public-api/Cargo.toml",
    ):
        assert f"--manifest-path {manifest}" in workflow


def test_fast_and_full_ci_keep_their_cost_boundary() -> None:
    fast = (ROOT / ".github/workflows/fast-ci.yml").read_text()
    full = (ROOT / ".github/workflows/ci.yml").read_text()
    fast_config = yaml.load(fast, Loader=yaml.BaseLoader)
    full_config = yaml.load(full, Loader=yaml.BaseLoader)

    for inexpensive_gate in (
        "Check source and packaged legal files",
        "Check CODEOWNERS policy and generated artifacts",
        "Check active workflow contracts",
        "ruff check",
        "ruff format --check",
        "python -m compileall",
        "cargo fmt --all -- --check",
    ):
        assert inexpensive_gate in fast

    for expensive_gate in (
        "cargo-deny",
        "cargo test --workspace",
        "crates/tests/public-api/Cargo.toml",
        "Application Tests",
        "Release Artifact Contract",
        "Application Wheel",
        "Platform Wheels",
        "Collector Data",
        "Prediction Regression",
        "Engine Golden Regression",
    ):
        assert expensive_gate in full
        assert expensive_gate not in fast

    assert "uses: ./.github/workflows/fast-ci.yml" not in full
    assert "workflow_call" not in fast_config["on"]
    assert fast_config["on"]["push"]["branches"] == full_config["on"]["push"]["branches"]
    prerequisite = full_config["jobs"]["fast-ci"]
    assert prerequisite["name"] == "Require Fast CI"
    assert prerequisite["needs"] == "verify-target"
    assert prerequisite["permissions"] == {"contents": "read", "actions": "read"}
    assert "uses" not in prerequisite
    prerequisite_steps = [
        step for step in prerequisite["steps"]
        if step.get("run") == "python scripts/require_fast_ci.py"
    ]
    assert len(prerequisite_steps) == 1
    assert prerequisite_steps[0]["env"]["GH_TOKEN"] == "${{ github.token }}"
    for job_id, job in full_config["jobs"].items():
        if job_id not in {"verify-target", "select-full-ci", "fast-ci", "stage-application-wheel"}:
            assert "fast-ci" in job["needs"], job_id
    assert "uses: ./.github/workflows/validate-platform-wheels.yml" in full
    assert "uses: ./.github/workflows/collector-check.yml" in full
    assert "uses: ./.github/workflows/prediction-regression-gate.yml" in full
    assert "name: Select Full CI Scope" in full
    assert "needs: [fast-ci, select-full-ci]" in full
    assert "EXPECTED_SHA: ${{ inputs.expected_sha }}" in full
    assert "RUN_SHA: ${{ github.sha }}" in full
    assert "expected_sha: ${{ github.sha }}" in full
    assert "needs: verify-target" in full
    assert "name: Fast CI Success" in fast
    assert "name: Full CI Success" in full
    assert "manual Full CI requires a nonempty expected_sha" in full
    assert (
        'if [[ -n "${EXPECTED_SHA}" && "${EXPECTED_SHA}" != "${RUN_SHA}" ]]; then'
        in full
    )
    assert "workflow_dispatch" in full_config["on"]
    dispatch_sha = full_config["on"]["workflow_dispatch"]["inputs"]["expected_sha"]
    assert dispatch_sha["required"] == "true"
    assert "default" not in dispatch_sha
    assert full_config["on"]["push"]["branches"] == [
        "main",
        "pull-request/*",
        "release/*",
    ]
    assert set(fast_config["on"]["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "labeled",
        "unlabeled",
    }

    fast_readiness = fast_config["jobs"]["readiness"]
    assert fast_readiness["name"] == "Fast CI Success"
    assert fast_readiness["if"] == "${{ always() }}"
    assert set(fast_readiness["needs"]) == {
        "policy",
        "python-static",
        "rust-format",
    }
    assert fast_readiness["env"] == {
        "IS_DRAFT": "${{ github.event.pull_request.draft || false }}",
        "HAS_REVIEW_READY": "${{ github.event_name != 'pull_request' || "
        "contains(github.event.pull_request.labels.*.name, 'review-ready') }}",
        "POLICY_RESULT": "${{ needs.policy.result }}",
        "PYTHON_STATIC_RESULT": "${{ needs.python-static.result }}",
        "RUST_FORMAT_RESULT": "${{ needs.rust-format.result }}",
    }
    fast_readiness_script = fast_readiness["steps"][0]["run"]
    assert "must have the review-ready label" in fast_readiness_script
    for result in (
        "${POLICY_RESULT}",
        "${PYTHON_STATIC_RESULT}",
        "${RUST_FORMAT_RESULT}",
    ):
        assert result in fast_readiness_script

    assert full_config["permissions"]["pull-requests"] == "read"
    verify_target = full_config["jobs"]["verify-target"]
    verify_copy_steps = [
        step for step in verify_target["steps"] if step.get("name") == "Verify trusted PR copy matches originating head"
    ]
    assert len(verify_copy_steps) == 1
    assert verify_copy_steps[0]["if"] == ("startsWith(github.ref, 'refs/heads/pull-request/')")
    assert "repos/${REPOSITORY}/pulls/${pr_number}" in verify_copy_steps[0]["run"]
    assert '"${pr_head}" != "${RUN_SHA}"' in verify_copy_steps[0]["run"]

    full_readiness = full_config["jobs"]["readiness"]
    assert full_readiness["name"] == "Full CI Success"
    assert full_readiness["if"] == "${{ always() }}"
    assert set(full_readiness["needs"]) == {
        "verify-target",
        "select-full-ci",
        "application-test-wheel",
        "fast-ci",
        "platform-wheels",
        "collector-data",
        "prediction-regression",
        "cargo-deny",
        "rust",
        "rust-feature-modes",
        "public-api-rust",
        "application-tests",
        "python-compatibility",
        "engine-golden-regression",
        "release-artifact-contract",
        "application-wheel",
    }
    assert set(full_readiness["needs"]) == set(full_config["jobs"]) - {"readiness", "stage-application-wheel"}
    full_readiness_script = full_readiness["steps"][0]["run"]
    full_readiness_env = full_readiness["steps"][0]["env"]
    assert full_readiness_env["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert '"success" if selected == "true" else "skipped"' in full_readiness_script
    assert (
        full_readiness_env["PLAN_JSON"] == "${{ toJSON(needs.select-full-ci.outputs) }}"
    )
    assert "stage-application-wheel" not in full_readiness["needs"]
    assert 'payload["result"]' in full_readiness_script

    assert set(full_config["jobs"]["stage-application-wheel"]["needs"]) == {"readiness", "application-wheel"}

    application_wheel = full_config["jobs"]["application-wheel"]
    assert "select-full-ci" in application_wheel["needs"]
    assert (
        "needs.select-full-ci.outputs.application_wheel == 'true'"
        in application_wheel["if"]
    )
    verify_steps = [
        step
        for step in application_wheel["steps"]
        if step.get("name") == "Verify exact staged wheel"
    ]
    assert len(verify_steps) == 1
    verify_step = verify_steps[0]
    assert verify_step["run"] == ("python python/aisimulate/tools/verify_release_wheels.py dist")
    assert "if" not in verify_step
    assert "continue-on-error" not in verify_step
    assert full_config["jobs"]["stage-application-wheel"]["if"] == (
        "github.event_name == 'push' && "
        "(github.ref == 'refs/heads/main' ||\n "
        "startsWith(github.ref, 'refs/heads/release/'))"
    )


def test_fast_ci_readiness_fails_closed(tmp_path: Path) -> None:
    config = yaml.load(
        (ROOT / ".github/workflows/fast-ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = config["jobs"]["readiness"]["steps"][0]["run"]
    passing = {
        "IS_DRAFT": "false",
        "HAS_REVIEW_READY": "true",
        "POLICY_RESULT": "success",
        "PYTHON_STATIC_RESULT": "success",
        "RUST_FORMAT_RESULT": "success",
    }

    assert _run_readiness_script(script, tmp_path, passing).returncode == 0

    missing_label = {**passing, "HAS_REVIEW_READY": "false"}
    assert _run_readiness_script(script, tmp_path, missing_label).returncode != 0

    skipped_job = {**passing, "PYTHON_STATIC_RESULT": "skipped"}
    assert _run_readiness_script(script, tmp_path, skipped_job).returncode != 0

    draft = {**passing, "IS_DRAFT": "true", "HAS_REVIEW_READY": "false"}
    assert _run_readiness_script(script, tmp_path, draft).returncode == 0


def test_full_ci_readiness_fails_closed(tmp_path: Path) -> None:
    config = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    readiness = config["jobs"]["readiness"]
    script = readiness["steps"][0]["run"]
    passing_results = {name: {"result": "success"} for name in readiness["needs"]}
    passing_pr = {
        "NEEDS_JSON": json.dumps(passing_results),
        "PLAN_JSON": json.dumps(
            {name: "true" for name in config["jobs"]["select-full-ci"]["outputs"]}
        ),
    }

    assert _run_readiness_script(script, tmp_path, passing_pr).returncode == 0

    canceled_results = {**passing_results, "rust": {"result": "cancelled"}}
    canceled_job = {**passing_pr, "NEEDS_JSON": json.dumps(canceled_results)}
    assert _run_readiness_script(script, tmp_path, canceled_job).returncode != 0

    failed_results = {**passing_results, "rust": {"result": "failure"}}
    failed_job = {**passing_pr, "NEEDS_JSON": json.dumps(failed_results)}
    assert _run_readiness_script(script, tmp_path, failed_job).returncode != 0

    missing_results = {
        **passing_results,
        "application-tests": {"result": ""},
    }
    missing_job = {**passing_pr, "NEEDS_JSON": json.dumps(missing_results)}
    assert _run_readiness_script(script, tmp_path, missing_job).returncode != 0


def test_full_ci_exact_target_verification(tmp_path: Path) -> None:
    config = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = {step["name"]: step for step in config["jobs"]["verify-target"]["steps"]}
    manual_script = steps["Reject a mismatched requested commit"]["run"]
    target_sha = "0123456789abcdef"

    matching = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "EXPECTED_SHA": target_sha,
        "RUN_SHA": target_sha,
    }
    assert _run_readiness_script(manual_script, tmp_path, matching).returncode == 0

    empty = {**matching, "EXPECTED_SHA": ""}
    assert _run_readiness_script(manual_script, tmp_path, empty).returncode != 0

    mismatched = {**matching, "EXPECTED_SHA": "fedcba9876543210"}
    assert _run_readiness_script(manual_script, tmp_path, mismatched).returncode != 0

    push_without_input = {
        **matching,
        "GITHUB_EVENT_NAME": "push",
        "EXPECTED_SHA": "",
    }
    assert _run_readiness_script(manual_script, tmp_path, push_without_input).returncode == 0


def test_full_ci_trusted_copy_verification(tmp_path: Path) -> None:
    config = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = {step["name"]: step for step in config["jobs"]["verify-target"]["steps"]}
    copy_script = steps["Verify trusted PR copy matches originating head"]["run"]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "exit_code=${FAKE_GH_EXIT:-0}\n"
        'if [[ ${exit_code} != 0 ]]; then exit "${exit_code}"; fi\n'
        "printf '%s\\t%s\\n' \"${FAKE_PR_HEAD:-}\" \"${FAKE_PR_BASE:-}\"\n"
    )
    fake_gh.chmod(0o755)
    target_sha = "a" * 40
    base_sha = "b" * 40
    matching = {
        "GITHUB_REF": "refs/heads/pull-request/135",
        "REPOSITORY": "ai-dynamo/aisimulate",
        "RUN_SHA": target_sha,
        "FAKE_PR_HEAD": target_sha,
        "FAKE_PR_BASE": base_sha,
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    assert _run_readiness_script(copy_script, tmp_path, matching).returncode == 0
    assert (tmp_path / "output.txt").read_text() == f"base_sha={base_sha}\n"

    mismatched = {**matching, "FAKE_PR_HEAD": "fedcba9876543210"}
    assert _run_readiness_script(copy_script, tmp_path, mismatched).returncode != 0

    invalid_ref = {**matching, "GITHUB_REF": "refs/heads/pull-request/not-a-number"}
    assert _run_readiness_script(copy_script, tmp_path, invalid_ref).returncode != 0

    api_failure = {**matching, "FAKE_GH_EXIT": "1"}
    assert _run_readiness_script(copy_script, tmp_path, api_failure).returncode != 0


def test_coderabbit_is_opted_in_by_review_ready_label() -> None:
    policy = yaml.safe_load((ROOT / ".coderabbit.yaml").read_text())
    auto_review = policy["reviews"]["auto_review"]

    assert auto_review["enabled"] is False
    assert auto_review["labels"] == ["review-ready", "!wip", "!do-not-review"]
    assert auto_review["drafts"] is False
    assert auto_review["base_branches"] == ["release/.*"]
    assert auto_review["ignore_title_keywords"] == [
        "WIP",
        "[skip review]",
        "[no review]",
    ]
    assert policy["reviews"]["request_changes_workflow"] is False
