# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid checkpoint geometry must reach a capable engine without being dropped."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import KvCacheRecommendationConfig
from aisimulate.runner import EngineReplayRunnerFactory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def config():
    return {
        "engine": {
            "mode": "aggregated",
            "backend": "sglang",
            "model": "synthetic-hybrid-cache",
            "hardware": "b200_sxm",
            "workers": {
                "aggregated": {
                    "scheduler": {"max_batched_tokens": 128},
                    "kv_cache": {
                        "block_size": 64,
                        "capacity": {"type": "fixed", "blocks": 128},
                        "mamba_cache": {"max_states": 64},
                    },
                    "timing": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


def test_checkpoint_config_survives_public_compilation_and_runner_gate():
    public = CorePredictionConfig.model_validate(config())
    public = CorePredictionConfig.model_validate(public.model_dump(mode="json"))
    spec = prediction_to_replay_spec(public)
    assert spec.backend_deployment.agg_engine_args["sglang"] == {
        "mamba_cache": {"max_states": 64, "prefill_checkpoint_interval": 64, "decode_checkpoint_interval": 256},
        "chunked_prefill_size": 128,
    }
    capabilities = EngineReplayRunnerFactory().capabilities()
    assert capabilities.supports_sglang_mamba_cache
    capabilities.require_compatible(spec)
    with pytest.raises(ValueError, match="does not support SGLang mamba_cache"):
        replace(capabilities, supports_sglang_mamba_cache=False).require_compatible(spec)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("max_states", 3, "greater than or equal to 4"),
        ("max_states", True, "valid integer"),
        ("prefill_checkpoint_interval", 0, "greater than 0"),
        ("prefill_checkpoint_interval", 32, "multiple of block_size"),
        ("decode_checkpoint_interval", 65, "multiple of prefill_checkpoint_interval"),
    ],
)
def test_invalid_checkpoint_geometry_is_rejected(field, value, error):
    raw = config()
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["mamba_cache"][field] = value
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_checkpoint_config_rejects_other_backends(backend):
    raw = config()
    raw["engine"]["backend"] = backend
    with pytest.raises(ValidationError, match="mamba_cache requires backend=sglang"):
        CorePredictionConfig.model_validate(raw)


def test_checkpoint_config_rejects_unsupported_dp_and_disabled_prefix():
    raw = config()
    worker = raw["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"attention_data": 4}
    with pytest.raises(ValidationError, match="mamba_cache currently requires attention_data=1"):
        CorePredictionConfig.model_validate(raw)
    worker.pop("parallelism")
    worker["kv_cache"]["prefix_caching"] = False
    with pytest.raises(ValidationError, match="mamba_cache requires prefix_caching=true"):
        CorePredictionConfig.model_validate(raw)


def test_checkpoint_config_rejects_recommendation():
    with pytest.raises(ValidationError, match="mamba_cache currently supports prediction only"):
        KvCacheRecommendationConfig.model_validate({"mamba_cache": {"max_states": 64}})
