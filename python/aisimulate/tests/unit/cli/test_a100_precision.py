# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject unsupported FP8 compute before starting replay."""

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["auto", "op_level", "fpm_interpolation"])
def test_a100_fp8_is_rejected_before_replay(mode):
    config = ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-32B-FP8",
        system="a100_sxm",
        backend="vllm",
        backend_version="current",
        worker_type="aggregated",
        estimation_mode=mode,
    )
    with pytest.raises(ValueError, match="a100_sxm.*does not support FP8.*fp8_block") as error:
        RustForwardPassPerfModel.best_available(config)
    assert "BF16 checkpoint" in str(error.value)
    assert "poisoned" not in str(error.value)


def test_a100_bf16_remains_ready():
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="Qwen/Qwen3-8B",
            system="a100_sxm",
            backend="vllm",
            backend_version="current",
            worker_type="aggregated",
        )
    )
    try:
        assert model.diagnostics()["readiness"] == "ready"
    finally:
        model.close()
