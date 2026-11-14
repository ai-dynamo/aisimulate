# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Modified behavioral regressions for SGLang; see THIRD_PARTY_NOTICES.md.
# Sources: v0.5.21 e00930c5489053f26d86b179cee0d087f846acbb (SWA),
# v0.5.22 baa9befa381511cd4d56f83980951e5a45fffc6d (DP budgets).
# https://github.com/sgl-project/sglang

"""SGLang PD replay: prefill chunk size, decode DP balancing and V4 transfer bytes."""

import json

import pytest

from aisim.capacity import estimate_kv_bytes_per_token
from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig
from aisim.runner import EngineReplayRunnerFactory

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


def _public(*, prefill=None, decode=None):
    prefill_worker = _worker(**(prefill or {}))
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


def test_nested_rank_load_balance_method_names_the_flat_field():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public()))
    decode = spec.backend_deployment.decode_engine_args
    decode["rank"] = {"load_balance_method": "total_tokens"}
    with pytest.raises(ValueError, match="flat engine argument, not a rank field"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(spec)


def test_unaligned_token_budget_chunks_whole_pages():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public(prefill={"max_batched_tokens": 2049})))
    assert spec.backend_deployment.prefill_engine_args["sglang"]["chunked_prefill_size"] == 2048


def test_deepseek_v4_transfer_bytes_follow_its_compressed_cache():
    from aisim_core.sdk.memory.deepseek_v4 import deepseek_v4_state_cache
    from aisim_core.sdk.models import _get_model_info

    expected = deepseek_v4_state_cache(_get_model_info(FLASH)["extra_params"], indexer_cache_dtype="auto")
    bytes_per_token = estimate_kv_bytes_per_token(FLASH, tp_size=1, pp_size=4)
    assert bytes_per_token == expected["kv_bytes_per_token"]
    # FP8 compressed KV plus indexer K, far below the dense BF16 formula (43 layers x 1024 x 2 bytes).
    assert bytes_per_token < 43 * 1024 * 2 // 10
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        estimate_kv_bytes_per_token(FLASH, tp_size=1, pp_size=1, kvcache_quant_mode="bfloat16")


def test_malformed_architectures_fail_loudly():
    from aisim_core.sdk.memory.kv_cache import NaiveKVCacheEstimator

    with pytest.raises(ValueError, match="architectures must be a list of strings"):
        NaiveKVCacheEstimator.from_hf_config({"architectures": "DeepseekV4ForCausalLM"}, tp_size=1, pp_size=1)


def test_unregistered_local_model_replays_with_fixed_timing_and_capacity(tmp_path):
    # The default SWA detection reads only the architecture; no performance model is needed.
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["CustomForCausalLM"]}))
    public = _public()
    public["engine"]["model"] = str(tmp_path)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(public))
    assert "swa_prefix_tails" not in spec.backend_deployment.prefill_engine_args["sglang"]
    EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(spec)


def _native(public, requests):
    from aisim import _runtime

    recording = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=recording).create(0).run(
        prediction_to_replay_spec(CorePredictionConfig.model_validate(public))
    )
    execution = recording.execution_spec["spec"]
    payload = {
        "version": 1,
        "topology": execution["topology"],
        "engine": execution["engine"],
        "adapters": execution["adapters"],
        "record_per_request": True,
        "requests": requests,
    }
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["completed_requests"] == len(requests)
    assert report["total_output_tokens"] == sum(r["output_tokens"] for r in requests)
    rows = {row["request_id"]: row for row in report["per_request"]}
    assert all(row["output_length"] == row["requested_output_length"] for row in rows.values())
    return rows


def _request(name, tokens, *, at=0.0, output=1, **fields):
    return {
        "id": name,
        "arrival_time_ms": at,
        "input_tokens": len(tokens),
        "input_token_ids": tokens,
        "output_tokens": output,
        **fields,
    }


@pytest.mark.parametrize("host_loop", [False, True])
@pytest.mark.parametrize("budget", [1, 2048, 2049])
def test_public_pd_chunk_alignment_reaches_native(host_loop, budget):
    public = _public(prefill={"max_batched_tokens": budget})
    worker = public["engine"]["workers"]["prefill"]
    worker["host_loop"] = host_loop
    worker["kv_cache"]["block_size"] = 16
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(public))
    assert spec.backend_deployment.prefill_engine_args["sglang"]["chunked_prefill_size"] == max(16, budget // 16 * 16)
    _native(public, [_request("prompt", list(range(65)))])


def _aggregated_swa(tails):
    public = _public()
    public["engine"].update(mode="aggregated", workers={"aggregated": _worker(max_sequences=1)})
    cache = public["engine"]["workers"]["aggregated"]["kv_cache"]
    cache.update(capacity={"type": "fixed", "blocks": 256}, swa_prefix_tails=tails)
    return public


@pytest.mark.parametrize("tails", [None, 8])
def test_public_swa_continuation_keeps_exact_generated_tail(tails):
    prompt = list(range(600))
    outputs = list(range(10_000, 10_425))
    rows = _native(
        _aggregated_swa(tails),
        [
            _request("seed", prompt, output=len(outputs), output_token_ids=outputs),
            _request("continuation", prompt + outputs + [20_000], at=1000.0),
        ],
    )
    assert rows["continuation"]["first_admission_reused_input_tokens"] == 1024


@pytest.mark.parametrize("hicache", [False, True])
def test_selective_hicache_does_not_inflate_unbacked_swa_room(hicache):
    public = _aggregated_swa(2)  # Three device SWA pages, independent of full KV pressure.
    if hicache:
        public["engine"]["workers"]["aggregated"]["kv_cache"].update(
            bytes_per_token=1024,
            hicache={"num_host_blocks": 1024, "write_policy": "write_through_selective"},
        )
    prompts = [list(range(i * 1000, i * 1000 + 257)) for i in range(5)]
    requests = [_request(str(i), tokens, at=i * 100.0) for i, tokens in enumerate(prompts)]
    requests += [_request("continuation", prompts[0] + [99_999], at=600.0)]
    rows = _native(public, requests)
    assert rows["continuation"]["first_admission_reused_input_tokens"] == 0
    if hicache:
        assert all(row["first_admission_host_reused_input_tokens"] == 0 for row in rows.values())
    else:
        assert all("first_admission_host_reused_input_tokens" not in row for row in rows.values())


def _token_balanced_public():
    public = _public(decode={"load_balance_method": "total_tokens"})
    worker = public["engine"]["workers"]["decode"]
    worker["parallelism"] = {"tensor": 2, "attention_data": 2}
    worker["kv_cache"]["capacity"]["blocks"] = 512
    public["engine"]["context_length"] = 16_384
    return public


def _decode_rank(row):
    return next(route["dp_rank"] for route in row["routing_history"] if route["pool"] == "decode")


def test_native_token_balance_mixed_output_budgets():
    rows = _native(
        _token_balanced_public(),
        [
            _request(str(i), list(range(prompt)), output=budget)
            for i, (prompt, budget) in enumerate([(1000, 1), (1, 10_000), (1, 1)])
        ],
    )
    assert [_decode_rank(rows[str(i)]) for i in range(3)] == [0, 1, 1]


def test_native_token_balance_refreshes_generated_kv_and_terminal_load():
    rows = _native(
        _token_balanced_public(),
        [
            _request("long-output", [1], output=300),
            _request("long-prompt", list(range(100)), at=125.0, output=200),
            # After 150 ms rank 0 holds more generated KV than rank 1.
            _request("after-growth", [2], at=150.0),
            # Both old requests have ended; fresh zero-load snapshots restore rank-order ties.
            _request("after-idle", [3], at=500.0),
        ],
    )
    keys = ("long-output", "long-prompt", "after-growth", "after-idle")
    assert [_decode_rank(rows[key]) for key in keys] == [0, 1, 1, 0]
