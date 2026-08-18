# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-specific routing contract for AISimulate CODEOWNERS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codeowners import (
    is_explicitly_owned,
    load_policy,
    owners_for,
    render,
    tracked_files,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_policy(ROOT / ".github/codeowners/areas.yaml")
MAINTAINERS = "@ai-dynamo/access-aisimulate-maintain"
FPE = "@ai-dynamo/aisimulate-forward-pass-engine-codeowners"
SWEEPER = "@ai-dynamo/aisimulate-sweeper-codeowners"
REPLAY = "@ai-dynamo/aisimulate-replay-codeowners"
MOCKER = "@ai-dynamo/aisimulate-mocker-codeowners"
INFRA = "@ai-dynamo/aisimulate-infra-codeowners"
DEVOPS = "@ai-dynamo/devops"


def _owners(path: str) -> set[str]:
    return set(owners_for(POLICY, path))


def test_representative_routing_contract() -> None:
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
    assert _owners(".github/workflows/ci.yml") == {INFRA}
    assert _owners(".github/workflows/codeowners.yml") == {DEVOPS}
    assert _owners(".github/codeowners/areas.yaml") == {DEVOPS}
    assert _owners("CODEOWNERS") == {DEVOPS}
    assert _owners("README.md") == {MAINTAINERS}


def test_all_tracked_paths_have_explicit_ownership() -> None:
    unowned = [
        path for path in tracked_files(ROOT) if not is_explicitly_owned(POLICY, path)
    ]
    assert unowned == []


def test_generated_file_matches_policy() -> None:
    assert (ROOT / "CODEOWNERS").read_text() == render(POLICY)


def test_unclassified_future_path_uses_maintainer_fallback() -> None:
    assert _owners("future/unclassified.txt") == {MAINTAINERS}
