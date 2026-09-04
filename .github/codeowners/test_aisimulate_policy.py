# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-specific routing contract for AISimulate's generated CODEOWNERS."""

import subprocess
import sys
from pathlib import Path

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
