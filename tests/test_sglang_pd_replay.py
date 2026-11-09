# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang PD replay: PP prefill, prefill chunk size, decode DP balancing and V4 transfer bytes."""

import json

import pytest

from aisimulate.capacity import estimate_kv_bytes_per_token, reject_pipeline_parallelism
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FLASH = "deepseek-ai/DeepSeek-V4-Flash"


class RecordingRuntime:
    execution_spec = None

    def run_replay_json(self, payload):
        self.execution_spec = json.loads(payload)
        return json.dumps({"duration_ms": 1.0, "completed_requests": 1})


def _worker(**scheduler):
    return {
        "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 64}},
        "scheduler": scheduler,
        "timing": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }


def _public(*, prefill=None, decode=None, prefill_pipeline=1):
    prefill_worker = _worker(**(prefill or {}))
    prefill_worker["parallelism"] = {"pipeline": prefill_pipeline}
    return {
        "engine": {
            "mode": "disaggregated",
            "backend": "sglang",
            "model": "sglang-pd-smoke",
            "hardware": "h200_sxm",
            "context_length": 4096,
            "workers": {"prefill": prefill_worker, "decode": _worker(**(decode or {}))},
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


def test_pd_prefill_chunks_at_max_batched_tokens_and_decode_balances_by_tokens():
    config = CorePredictionConfig.model_validate(
        _public(
            prefill={"max_batched_tokens": 2048},
            decode={"load_balance_method": "total_tokens"},
        )
    )
    spec = prediction_to_replay_spec(config)
    assert spec.backend_deployment.prefill_engine_args["sglang"]["chunked_prefill_size"] == 2048
    assert spec.backend_deployment.decode_engine_args["load_balance_method"] == "total_tokens"
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    execution = runtime.execution_spec["spec"]
    assert execution["adapters"]["placement"]["config"] == {"load_balance_method": "total_tokens"}
    assert "load_balance_method" not in json.dumps(execution["engine"])


def test_round_robin_decode_keeps_the_default_placement():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public()))
    assert "load_balance_method" not in spec.backend_deployment.decode_engine_args
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec["spec"]["adapters"]["placement"]["config"] is None


def test_load_balance_method_is_a_disaggregated_decode_setting():
    with pytest.raises(ValueError, match="only for the disaggregated decode role"):
        CorePredictionConfig.model_validate(_public(prefill={"load_balance_method": "total_tokens"}))
    aggregated = _public()
    aggregated["engine"]["mode"] = "aggregated"
    aggregated["engine"]["workers"] = {"aggregated": _worker(load_balance_method="total_tokens")}
    with pytest.raises(ValueError, match="only for the disaggregated decode role"):
        CorePredictionConfig.model_validate(aggregated)


def test_replay_spec_rejects_load_balance_method_outside_decode():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public()))
    spec.backend_deployment.prefill_engine_args["load_balance_method"] = "total_tokens"
    with pytest.raises(ValueError, match="applies only to the decode role"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(spec)


def test_unaligned_token_budget_chunks_whole_pages():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public(prefill={"max_batched_tokens": 2049})))
    assert spec.backend_deployment.prefill_engine_args["sglang"]["chunked_prefill_size"] == 2048


def test_sglang_pd_prefill_accepts_pipeline_stages():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public(prefill_pipeline=4)))
    assert spec.backend_deployment.parallel_config["prefill_pp"] == 4


@pytest.mark.parametrize(
    ("role", "backend"),
    [
        ("prefill", "vllm"),
        ("decode", "sglang"),
        ("aggregated", "sglang"),
        ("prefill", None),
    ],
)
def test_pipeline_parallelism_outside_sglang_pd_prefill_is_rejected(role, backend):
    with pytest.raises(ValueError, match=f"does not support pipeline parallelism for {role}"):
        reject_pipeline_parallelism(role, 2, backend=backend)
    reject_pipeline_parallelism("prefill", 2, backend="sglang")


def test_deepseek_v4_transfer_bytes_follow_its_compressed_cache():
    from aisimulate_core.sdk.memory.deepseek_v4 import deepseek_v4_state_cache
    from aisimulate_core.sdk.models import _get_model_info

    expected = deepseek_v4_state_cache(_get_model_info(FLASH)["extra_params"], indexer_cache_dtype="auto")
    bytes_per_token = estimate_kv_bytes_per_token(FLASH, tp_size=1, pp_size=4)
    assert bytes_per_token == expected["kv_bytes_per_token"]
    # FP8 compressed KV plus indexer K, far below the dense BF16 formula (43 layers x 1024 x 2 bytes).
    assert bytes_per_token < 43 * 1024 * 2 // 10
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        estimate_kv_bytes_per_token(FLASH, tp_size=1, pp_size=1, kvcache_quant_mode="bfloat16")


def test_malformed_architectures_fail_loudly():
    from aisimulate_core.sdk.memory.kv_cache import NaiveKVCacheEstimator

    with pytest.raises(ValueError, match="architectures must be a list of strings"):
        NaiveKVCacheEstimator.from_hf_config({"architectures": "DeepseekV4ForCausalLM"}, tp_size=1, pp_size=1)
