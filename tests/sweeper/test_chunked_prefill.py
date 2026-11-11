# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Guangli Liu
# SPDX-License-Identifier: Apache-2.0

"""Explicit SGLang chunks survive search, saved prediction and native replay."""

import pytest

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.search_space import branch_knob_choices

pytestmark = pytest.mark.unit


def _config(role="aggregated", chunk=None):
    roles = ("aggregated",) if role == "aggregated" else ("prefill", "decode")
    workers = {
        name: {
            "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
            "kv_cache": {"block_size": 1, "capacity": {"type": "fixed", "blocks": 8192}},
            "timing": {"type": "fixed", "prefill_ms": 10, "decode_ms": 1},
        }
        for name in roles
    }
    if chunk is not None:
        workers[role]["scheduler"]["chunked_prefill_size"] = chunk
    return {
        "engine": {
            "model": "Qwen/Qwen3-32B",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.22",
            "context_length": 4096,
            "mode": "aggregated" if role == "aggregated" else "disaggregated",
            "workers": workers,
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 256, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    "domain,expected,bounds",
    [
        (128, None, None),
        ({"choices": [64, 128, 256]}, [64, 128, 256], None),
        ({"range": {"min": 64, "max": 256, "step": 64}}, [64, 128, 192, 256], None),
        ({"range": {"min": 64, "max": 256, "scale": "log"}}, [64], [64, 256]),
    ],
)
def test_chunk_domain_round_trip(role, domain, expected, bounds):
    source = CoreRecommendationConfig.model_validate({**_config(role, domain), "optimization": {}})
    smart = recommendation_to_sweeper(source)
    space = smart.search_space
    mode = "agg" if role == "aggregated" else "disagg"
    internal_role = "agg" if role == "aggregated" else role
    key = f"{internal_role}_chunked_prefill_size"
    knobs = branch_knob_choices(space, mode)
    if expected is None:
        assert key not in knobs
        assert getattr(space, key) == 128
    else:
        assert knobs[key] == expected
    assert space.engine_integer_log_ranges.get(key) == bounds
    replica = ReplicaParallelConfig(ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=1), replicas=1)
    parallel = replica if mode == "agg" else DisaggParallelConfig(replica, replica)
    selection = {"deployment_mode": mode, "backend": "sglang"}
    for name in ("agg",) if mode == "agg" else ("prefill", "decode"):
        selection[f"{name}_max_num_batched_tokens"] = 8192
        selection[f"{name}_max_num_seqs"] = 8
    if expected is not None:
        selection[key] = 128
    sample = unroll_sample(search_space=space, selection=selection, parallel_config=parallel)
    assert sample[key] == 128
    deployment = build_backend_deployment(sample, backend_version="0.5.22")
    args = getattr(deployment, f"{internal_role}_engine_args")
    # Keep the global chunk intact: the native scheduler divides it by DP once.
    assert args["aic_attention_dp_size"] == 2
    assert args["sglang"]["chunked_prefill_size"] == 128
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(mode="json"), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    assert saved["engine"]["workers"][role]["scheduler"]["chunked_prefill_size"] == 128
    reloaded = CorePredictionConfig.model_validate(saved)
    replay = prediction_to_replay_spec(reloaded)
    assert getattr(replay.backend_deployment, f"{internal_role}_engine_args")["sglang"]["chunked_prefill_size"] == 128


@pytest.mark.parametrize(
    "value", [0, -1, True, 1.5, {"choices": []}, {"choices": [64, 0]}, {"range": {"min": 0, "max": 64, "step": 32}}]
)
def test_recommend_rejects_invalid_chunks(value):
    with pytest.raises(ValueError, match="chunked_prefill_size"):
        CoreRecommendationConfig.model_validate({**_config(chunk=value), "optimization": {}})


@pytest.mark.parametrize("backend", ["vllm", "trtllm", {"choices": ["sglang", "vllm"]}])
def test_recommend_rejects_chunks_with_other_backends(backend):
    raw = _config(chunk=128)
    raw["engine"]["backend"] = backend
    with pytest.raises(ValueError, match="chunked_prefill_size.*sglang"):
        CoreRecommendationConfig.model_validate({**raw, "optimization": {}})


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_predict_rejects_chunks_with_other_backends(backend):
    raw = _config(chunk=128)
    raw["engine"]["backend"] = backend
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["block_size"] = 64
    with pytest.raises(ValueError, match="chunked_prefill_size.*sglang"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("value", [[], [0], -1, True, 1.5])
def test_internal_search_space_rejects_invalid_chunks(value):
    with pytest.raises(ValueError, match="chunked_prefill_size"):
        SearchSpace(
            model_name="example/model", hardware_sku="example", backend=["sglang"], agg_chunked_prefill_size=value
        )


@pytest.mark.parametrize("chunk,expected", [(None, None), (128, 128)])
def test_prediction_preserves_omitted_chunk(chunk, expected):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_config(chunk=chunk)))
    assert spec.backend_deployment.agg_engine_args.get("sglang", {}).get("chunked_prefill_size") == expected


@pytest.mark.parametrize("chunk,expected", [(None, 8192), (128, 128)])
def test_explicit_chunk_overrides_host_loop_fallback(chunk, expected):
    raw = _config(chunk=chunk)
    worker = raw["engine"]["workers"]["aggregated"]
    worker["host_loop"] = True
    worker.pop("timing")  # Host-loop replay requires native timing.
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert spec.backend_deployment.agg_engine_args["sglang"]["chunked_prefill_size"] == expected


@pytest.mark.parametrize("dp", [1, 2])
def test_chunk_controls_native_prefill_passes(dp):
    observed = []
    for chunk in (64 * dp, 128 * dp, 256 * dp):
        raw = _config(chunk=chunk)
        raw["engine"]["workers"]["aggregated"]["parallelism"] = {"attention_data": dp}
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        report = EngineReplayRunnerFactory().create(0).run(spec)
        assert report.metrics["completed_requests"] == 1
        observed.append(report.metrics["mean_ttft_ms"])
    # One 256-token prompt, fixed 10 ms per prefill pass: four, two, one passes.
    assert observed == pytest.approx([40, 20, 10])
