# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--ops`` naming an op the collector v2 case plan does not contain exits with the
dedicated EXIT_CODE_OPS_NOT_IN_PLAN, not argparse's 2, so a shard scheduler can mark
the shard terminal (sm100 shard plans 2026-10-05: the registry-only vllm
dsv4_hca_attn_module retried to max_attempts on eight systems). That op was removed
on 2026-10-15, so the probe below uses an op another backend's registry owns: a valid
``--ops`` choice that vLLM's case plan never carries, the shape of a stale or
cross-framework shard plan."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PKG_ROOT = Path(__file__).resolve().parents[3]


def _run_collect(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    return subprocess.run(
        [sys.executable, str(PKG_ROOT / "collector" / "collect.py"), *args],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_op_outside_the_backend_plan_exits_with_the_dedicated_code():
    from collector.helper import EXIT_CODE_OPS_NOT_IN_PLAN

    # mamba2 is a trtllm registry op: accepted by --ops, never in vLLM's case plan.
    result = _run_collect("--backend", "vllm", "--model-cases-full", "--plan-only", "--ops", "mamba2")

    assert result.returncode == EXIT_CODE_OPS_NOT_IN_PLAN, result.stderr[-2000:]
    assert EXIT_CODE_OPS_NOT_IN_PLAN not in (0, 1, 2)
    assert "Requested ops are not present in the collector v2 case plan: mamba2" in result.stderr


def test_planned_op_still_passes_the_ops_check():
    result = _run_collect("--backend", "vllm", "--model-cases-full", "--plan-only", "--ops", "gemm")

    assert result.returncode == 0, result.stderr[-2000:]
