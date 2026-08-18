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
DEVOPS = "@ai-dynamo/devops"
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
    assert _owners("crates/aisimulate-core/src/fpm/model.rs") == {
        FPE,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate-core/src/aisimulate_core/sdk/engine.py") == {
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

    # Standalone Replay, Sweeper, and Mocker surface.
    assert _owners("src/aisimulate/aic.py") == {FPE, MAINTAINERS}
    assert _owners("src/aisimulate/sweeper/search.py") == {SWEEPER, MAINTAINERS}
    assert _owners("src/aisimulate/replay/cli.py") == {REPLAY, MAINTAINERS}
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
    assert _owners("crates/python/src/lib.rs") == {
        MOCKER,
        REPLAY,
        FPE,
        MAINTAINERS,
    }
    assert _owners("crates/core/Cargo.toml") == {
        MOCKER,
        REPLAY,
        INFRA,
        MAINTAINERS,
    }

    # Active and imported repository metadata.
    assert _owners(".github/workflows/ci.yml") == {INFRA}
    assert _owners("scripts/build_release_artifacts.py") == {INFRA, MAINTAINERS}
    assert _owners("python/aisimulate/.github/workflows/build-test.yml") == {
        INFRA,
        MAINTAINERS,
    }
    assert _owners("python/aisimulate/pyproject.toml") == {
        FPE,
        INFRA,
        MAINTAINERS,
    }
    assert _owners(".github/workflows/codeowners.yml") == {DEVOPS}
    assert _owners(".github/codeowners/areas.yaml") == {DEVOPS}
    assert _owners("python/aisimulate/.github/codeowners/areas.yaml") == {DEVOPS}
    assert _owners("python/aisimulate/CODEOWNERS") == {DEVOPS}
    assert _owners("crates/aisimulate-core/deny.toml") == {DEVOPS}
    assert _owners("CODEOWNERS") == {DEVOPS}
    assert _owners("README.md") == {MAINTAINERS}


def test_unclassified_future_path_uses_maintainer_fallback() -> None:
    assert _owners("future/unclassified.txt") == {MAINTAINERS}
