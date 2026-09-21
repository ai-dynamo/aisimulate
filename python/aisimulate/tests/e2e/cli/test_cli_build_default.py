# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess as sp
import tempfile

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.build]

_DEFAULT_BUILD_CASES = [
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "trtllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
            "save_dir": tempfile.gettempdir(),
        },
    ),
    pytest.param(
        {
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system": "h200_sxm",
            "backend": "trtllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "sglang",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
        },
    ),
    pytest.param(
        {
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system": "h200_sxm",
            "backend": "sglang",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
        marks=pytest.mark.timeout(300),
    ),
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
        },
    ),
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "total_gpus": 1,  # agg only
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-8B",
            "system": "b60",
            "backend": "vllm",
            "total_gpus": 4,
            "isl": 1500,
            "osl": 150,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
    pytest.param(
        {
            "model_path": "Qwen/Qwen3-30B-A3B",
            "system": "b60",
            "backend": "vllm",
            "total_gpus": 8,
            "isl": 1500,
            "osl": 150,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
]


def _build_default_cmd(
    *,
    model_path: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    prefix: int,
    ttft: int,
    tpot: int,
    save_dir: str | None = None,
):
    cmd = [
        "aiconfigurator",
        "cli",
        "default",
        "--model-path",
        model_path,
        "--system",
        system,
        "--backend",
        backend,
        "--total-gpus",
        str(total_gpus),
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--prefix",
        str(prefix),
        "--ttft",
        str(ttft),
        "--tpot",
        str(tpot),
    ]
    if save_dir:
        cmd.extend(["--save-dir", save_dir])
    return cmd


@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", _DEFAULT_BUILD_CASES)
def test_cli_default_build_subset(case: dict):
    """
    Small, stable E2E subset for the GitHub PR workflow.

    This mirrors the previously hard-coded CI build selection (a small matrix of
    model/system/backend combinations) while using the reorganized CLI E2E tests.
    """

    cmd = _build_default_cmd(**case)

    completed = sp.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        combined = f"{completed.stdout}\n{completed.stderr}".strip()
        raise AssertionError(f"CLI default failed:\n{combined}")

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "AIConfigurator Final Results" in combined_output
    assert f"Model: {case['model_path']}" in combined_output
    assert f"Total GPUs: {case['total_gpus']}" in combined_output

    # TODO: remove try/except around save_results
    assert "Failed to save results" not in combined_output


@pytest.mark.parametrize("total_gpus", [4, 32])
def test_v41_sol_default_accepts_automatic_moe_parallelism(total_gpus):
    """Exercise the actual default sweep: model construction alone missed this bug."""
    completed = sp.run(
        [
            "aiconfigurator",
            "cli",
            "default",
            "--model",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "--system",
            "gb300",
            "--backend",
            "sglang",
            "--total-gpus",
            str(total_gpus),
            "--serving-mode",
            "agg",
            "--database-mode",
            "SOL",
            "--isl",
            "1024",
            "--osl",
            "128",
            "--nextn",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    # The CLI currently logs sweep failures without a nonzero exit code, so
    # require actual ranked results and the correct topology, not exit zero.
    assert "AIConfigurator Final Results" in output, output
    assert "Model: deepseek-ai/DeepSeek-V4.1-Flash (is_moe: True)" in output
    assert f"Total GPUs: {total_gpus}" in output
    assert "tp4pp1dp1etp1ep4" in output
    # Rendered GPU accounting is total (used = replicas x GPUs per replica).
    assert f"{total_gpus} ({total_gpus}={total_gpus // 4}x4)" in output
    assert "Parallelism width mismatch" not in output
    assert "No successful experiment runs" not in output
