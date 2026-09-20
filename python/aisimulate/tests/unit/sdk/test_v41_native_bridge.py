# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


def _native_model(config):
    return RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig.from_legacy_engine_config(config, "aggregated")
    )


def _config(replay=False, backend="sglang"):
    return {
        "schema_version": 1,
        "model_name": MODEL_PATH,
        "system_name": "gb300",
        "backend": backend,
        "backend_version": "0.5.14" if backend == "sglang" else "0.24.0",
        "tp_size": 4,
        "pp_size": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "database_mode": "SOL",
        "decoder_replay": replay,
    }


def test_native_builder_preserves_replay_and_backend_guard():
    sample = {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": 1,
            "sum_prefill_tokens": 256,
            "sum_prefill_kv_tokens": 0,
        },
    }
    full = _native_model(_config(False))
    bounded = _native_model(_config(True))
    assert 0 < bounded.estimate_forward_pass_time_ms(sample) < full.estimate_forward_pass_time_ms(sample)
    with pytest.raises(ValueError, match="decoder_replay requires"):
        _native_model(_config(True, "vllm"))


@pytest.mark.parametrize(
    ("model_path", "backend"),
    [("example/model", "sglang"), (MODEL_PATH, "vllm")],
)
def test_compile_engine_rejects_unsupported_decoder_replay(model_path, backend):
    from aisimulate_core.sdk.engine import compile_engine

    with pytest.raises(ValueError, match="decoder_replay requires"):
        compile_engine(model_path, "gb300", backend, decoder_replay=True)


@pytest.mark.parametrize("extends", [((1, 1023), (1023, 1)), ((512, 512), (512, 512)), ((1, 1023), (2, 1022))])
def test_replay_fpm_rejects_multiple_prefills_even_with_equal_prompts(extends):
    # FPM v1 reports variance of full prompt lengths, not actual extends.
    # All three valid batches have equal 1024-token prompts. Their aggregate
    # does not identify the individual tails, even if the caller knows them.
    assert {query + prefix for query, prefix in extends} == {1024}
    sample = {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": len(extends),
            "sum_prefill_tokens": sum(query for query, _ in extends),
            "sum_prefill_kv_tokens": sum(prefix for _, prefix in extends),
            "var_prefill_length": 0.0,
        },
    }
    bounded = _native_model(_config(True))
    with pytest.raises(ValueError, match="multiple prefill requests"):
        bounded.estimate_forward_pass_time_ms(sample)
    # Decoder OFF retains the existing aggregate approximation.
    full = _native_model(_config(False))
    assert full.estimate_forward_pass_time_ms(sample) > 0


@pytest.mark.parametrize("query", [1, 127, 128, 129])
def test_replay_fpm_single_prefill_matches_explicit_static_geometry(query):
    from aisimulate_core.sdk.engine import EngineHandle

    bounded = _native_model(_config(True))
    static = EngineHandle.compile(
        MODEL_PATH,
        "gb300",
        "sglang",
        backend_version="0.5.14",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        decoder_replay=True,
        database_mode="SOL",
    )
    sample = {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": 1,
            "sum_prefill_tokens": query,
            "sum_prefill_kv_tokens": 1024,
        },
    }
    assert bounded.estimate_forward_pass_time_ms(sample) == pytest.approx(
        static.predict_prefill_latency(1, 1024 + query, 1024), rel=1e-12
    )


def test_compile_engine_resolves_v41_native_expert_lane(monkeypatch):
    from aisimulate_core.sdk import engine

    original = engine.get_model
    captured = []

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        captured.append(model.config.moe_quant_mode.name)
        return model

    monkeypatch.setattr(engine, "get_model", build)
    handle = engine.EngineHandle.compile(
        MODEL_PATH,
        "gb300",
        "sglang",
        backend_version="0.5.14",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        database_mode="SOL",
    )
    assert captured == ["w4a8_mxfp4_mxfp8_trtllm"]
    assert handle.predict_prefill_latency(1, 256) > 0
