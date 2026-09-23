# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the existing trusted-PR selector for the complete simulation call path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_forward_perf

FILES = {
    ".github/workflows/simulation-performance.yml",
    "scripts/select_simulation_perf.py",
    "scripts/select_forward_perf.py",
    "Cargo.toml",
    "Cargo.lock",
    "crates/core/Cargo.toml",
    "python/aisimulate/pyproject.toml",
    "python/aisimulate/uv.lock",
}
PREFIXES = (
    "crates/core/src/",
    "python/aisimulate/src/aisimulate/",
    "python/aisimulate/src/aisimulate_core/",
    "python/aisimulate/tools/simulation_perf_gate/",
)


def matches_path(path: str) -> bool:
    if path.endswith(".md"):
        return False
    return path in FILES or path.startswith(PREFIXES)


def select_comparison(*args, **kwargs) -> dict:
    result = select_forward_perf.select_comparison(*args, path_matcher=matches_path, **kwargs)
    result["reason"] = result["reason"].replace("forward prediction", "simulation")
    return result


if __name__ == "__main__":
    raise SystemExit(select_forward_perf.main(selector=select_comparison, title="Simulation Performance"))
