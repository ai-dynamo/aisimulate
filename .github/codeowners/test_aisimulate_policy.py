# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-specific routing contract for AISimulate's generated CODEOWNERS."""

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


def test_subsystem_teams_retain_maintainer_coownership() -> None:
    rules = parse_codeowners((ROOT / "CODEOWNERS").read_text())
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    ).splitlines()
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
    full_config = yaml.load(full, Loader=yaml.BaseLoader)

    for inexpensive_gate in (
        "Check source and packaged legal files",
        "Check CODEOWNERS policy and generated artifacts",
        "ruff check",
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
    ):
        assert expensive_gate in full
        assert expensive_gate not in fast

    assert "uses: ./.github/workflows/fast-ci.yml" in full
    assert "needs: fast-ci" in full
    assert 'EXPECTED_SHA: ${{ inputs.expected_sha }}' in full
    assert 'RUN_SHA: ${{ github.sha }}' in full
    assert 'expected_sha: ${{ github.sha }}' in full
    assert "needs: verify-target" in full
    assert 'if [[ -n "${EXPECTED_SHA}" && "${EXPECTED_SHA}" != "${RUN_SHA}" ]]; then' in full
    assert "workflow_dispatch" in full_config["on"]
    assert full_config["on"]["push"]["branches"] == ["main", "release/*"]
    application_wheel = full_config["jobs"]["application-wheel"]
    assert "if" not in application_wheel
    verify_steps = [
        step
        for step in application_wheel["steps"]
        if step.get("name") == "Verify exact staged wheel"
    ]
    assert len(verify_steps) == 1
    verify_step = verify_steps[0]
    assert verify_step["run"] == (
        "python python/aisimulate/tools/verify_release_wheels.py dist"
    )
    assert "if" not in verify_step
    assert "continue-on-error" not in verify_step
    assert full_config["jobs"]["stage-application-wheel"]["if"] == (
        "github.event_name == 'push' && "
        "(github.ref == 'refs/heads/main' ||\n "
        "startsWith(github.ref, 'refs/heads/release/'))"
    )


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
