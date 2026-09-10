# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH
from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

pytestmark = pytest.mark.unit


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
    full = RustForwardPassPerfModel.from_native(_config(False))
    bounded = RustForwardPassPerfModel.from_native(_config(True))
    assert 0 < bounded.estimate_forward_pass_time_ms(sample) < full.estimate_forward_pass_time_ms(sample)
    with pytest.raises(Exception, match="not verified"):
        RustForwardPassPerfModel.from_native(_config(True, "vllm"))


def test_compile_engine_resolves_v41_native_expert_lane(monkeypatch):
    from aiconfigurator_core.sdk import engine

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
