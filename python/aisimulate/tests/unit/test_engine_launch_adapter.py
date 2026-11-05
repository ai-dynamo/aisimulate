# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisimulate.runner import canonical_performance_config, materialize_engine_launch_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_runner_exports_canonical_engine_launch(backend):
    config = materialize_engine_launch_config(
        backend,
        "",
        {},
        {
            "block_size": 16,
            "num_gpu_blocks": 64,
            "max_model_len": 128,
            "dp_size": 2,
            "tensor_parallel_size": 2,
            "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
        },
        "decode",
    )
    assert config["engine"]["backend"] == backend
    assert config["engine"]["worker_type"] == "decode"
    assert config["engine"]["max_model_len"] == 128
    assert config["dp_size"] == 2
    assert config["tensor_parallel_size"] == 2
    assert config["num_gpu_blocks_is_explicit"] is True
    assert "engine_type" not in config


def test_canonical_timing_metadata_preserves_controls_and_role():
    raw = {
        "model_path": "model",
        "system": "system",
        "backend": "vllm",
        "tp_size": 2,
        "attention_dp_size": 2,
        "pp_size": 1,
        "worker_type": "prefill",
        "estimator_config": {"correction": {"enabled": False}},
    }
    config = canonical_performance_config(raw, worker_type="prefill")
    assert config["model"] == "model"
    assert config["tp"] == 2
    assert config["attention_dp"] == 2
    assert config["estimator_config"]["correction"]["enabled"] is False
    with pytest.raises(ValueError, match="worker_type"):
        canonical_performance_config(raw, worker_type="decode")
