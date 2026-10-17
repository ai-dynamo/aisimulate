# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

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


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
def test_generic_speculation_survives_canonical_launch(tmp_path, backend, role, kind):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 4,
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
            }
        )
    )
    version = "0.24.0" if backend == "vllm" else "0.5.14"
    selection = {
        "kind": kind,
        "params": {"depth": 3} if kind == "mtp" else {"num_speculative_tokens": 3},
        "expected_accepted_tokens": 2.4,
        "seed": 73,
    }
    if kind == "eagle3":
        selection["draft_config"] = {
            "num_hidden_layers": 1,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 32000,
        }
    config = materialize_engine_launch_config(
        backend,
        version,
        {},
        {
            "block_size": 16,
            "num_gpu_blocks": 64,
            "speculation": selection,
            "timing_model": {
                "type": "external",
                "provider": "aic",
                "config": {
                    "model": str(target),
                    "backend_version": version,
                    "system": "h200_sxm",
                    "backend": backend,
                    "worker_type": role,
                    "estimation_mode": "op_level",
                },
            },
        },
        role,
    )
    rank = config["engine"]
    assert rank["aic_nextn"] == 3
    assert rank["aic_mtp_seed"] == 73
    assert [float(rate) for rate in rank["aic_nextn_accept_rates"].split(",")] == pytest.approx([1, 1, 0.4])
    cost = rank["timing_model"]["config"]
    assert cost["speculation"]["kind"] == kind
    assert cost["nextn"] == 0
    assert cost["worker_type"] == role
    assert config["num_gpu_blocks_is_explicit"] is True
