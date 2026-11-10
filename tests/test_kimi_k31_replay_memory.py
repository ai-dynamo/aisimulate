# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K31 cache requirements survive public configuration and native replay.

The checkpoint extends the bundled K3 fixture, as the model unit tests do.
Manual state sizes below are synthetic scheduler inputs, not native vLLM
allocation measurements or automatic physical-state qualification.
"""

import json
from copy import deepcopy

import pytest

from aisimulate import _runtime
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate_core.sdk.utils import get_model_config_from_model_path

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

# 24 replicated DSA layers, each with a 656-byte packed latent and a
# 128-byte FP8 index key plus one 4-byte scale. Fixed KDA state is additional.
TOKEN_BYTES = 18_912
BLOCK_BYTES = 64 * TOKEN_BYTES


@pytest.fixture
def k31_checkpoint(tmp_path):
    raw = deepcopy(get_model_config_from_model_path("moonshotai/Kimi-K3")["raw_config"])
    raw["architectures"] = ["KimiK31ForConditionalGeneration"]
    raw["model_type"] = "kimi_k31"
    raw["text_config"]["dsa_config"] = {
        "dsa_layers": list(raw["text_config"]["linear_attn_config"]["full_attn_layers"]),
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "index_topk_freq": 1,
        "index_skip_topk_offset": 1,
    }
    # No model-family substring in the path: the guard must use resolved
    # checkpoint/compiled architecture, not the caller's model name.
    checkpoint = tmp_path / "renamed-hybrid"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(raw))
    return str(checkpoint)


def _payload(checkpoint, *, state_bytes=None):
    cache = {
        "block_size": 64,
        "bytes_per_token": TOKEN_BYTES,
        "capacity": {"type": "fixed", "blocks": 64},
    }
    if state_bytes is not None:
        cache["state_cache"] = {"bytes_per_request": state_bytes}
    return {
        "engine": {
            "mode": "aggregated",
            "backend": "vllm",
            "model": checkpoint,
            "hardware": "gb300",
            "context_length": 1024,
            "database_mode": "SOL",
            "kvcache_quant_mode": "fp8",
            "attention_backend": "flashmla",
            "workers": {
                "aggregated": {
                    "parallelism": {"tensor": 8, "moe_tensor": 1, "moe_expert": 8},
                    "timing": {"type": "default"},
                    "kv_cache": cache,
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 32, "output_tokens": 2},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 2},
        },
    }


def _saved_spec(payload):
    config = CorePredictionConfig.model_validate(payload)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    return prediction_to_replay_spec(restored)


class _WireCapture:
    def run_replay_json(self, payload):
        self.payload = payload
        return json.dumps({"duration_ms": 1.0, "completed_requests": 0})


@pytest.mark.parametrize("direct_native", [False, True])
@pytest.mark.parametrize("host_offload", [False, True])
def test_k31_replay_requires_state_even_with_fixed_token_geometry(k31_checkpoint, direct_native, host_offload):
    payload = _payload(k31_checkpoint)
    if host_offload:
        payload["engine"]["workers"]["aggregated"]["kv_cache"]["host_offload"] = {"num_host_blocks": 100}
    spec = _saved_spec(payload)
    if direct_native:
        capture = _WireCapture()
        EngineReplayRunnerFactory(runtime=capture).create(0).run(spec)
        # Actually execute the saved wire through Rust, bypassing the Python
        # materializer. The capture itself is not a validation result.
        with pytest.raises(RuntimeError, match="requires explicit state_cache geometry"):
            _runtime.run_replay_json(capture.payload)
    else:
        with pytest.raises(RuntimeError, match="requires explicit state_cache geometry"):
            EngineReplayRunnerFactory().create(0).run(spec)


def test_k31_explicit_manual_state_survives_reload_and_native_execution(k31_checkpoint):
    spec = _saved_spec(_payload(k31_checkpoint, state_bytes=2 * BLOCK_BYTES))
    args = spec.backend_deployment.agg_engine_args
    assert args["kv_cache_bytes_per_token"] == TOKEN_BYTES
    assert args["state_cache"] == {"bytes_per_request": 2 * BLOCK_BYTES}
    metrics = EngineReplayRunnerFactory().create(0).run(spec).metrics
    assert metrics["completed_requests"] == 2
    assert metrics["total_input_tokens"] == 64
    assert metrics["total_output_tokens"] == 4


def test_k31_manual_pd_transfer_charges_one_state_per_request(k31_checkpoint):
    def ttft(state_bytes):
        payload = _payload(k31_checkpoint, state_bytes=state_bytes)
        engine = payload["engine"]
        worker = engine["workers"]["aggregated"]
        engine["mode"] = "disaggregated"
        engine["workers"] = {"prefill": worker, "decode": deepcopy(worker)}
        engine["kv_transfer"] = {"bytes_per_token": "auto", "bandwidth_gb_per_second": 1e-6}
        payload["traffic"]["stop"]["requests"] = 1
        spec = _saved_spec(payload)
        assert spec.backend_deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == TOKEN_BYTES
        result = EngineReplayRunnerFactory().create(0).run(spec).metrics
        assert result["completed_requests"] == 1
        return result["mean_ttft_ms"]

    # The only changed input is one additional state block. At 1000 B/s,
    # each additional byte contributes one millisecond, once per request.
    assert ttft(2 * BLOCK_BYTES) - ttft(BLOCK_BYTES) == pytest.approx(BLOCK_BYTES)
