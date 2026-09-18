# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine/request controls survive canonical construction and saved replay."""

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.common import ENGINE_MODEL_CONTROL_FIELDS
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import _accept_rates_for_expected
from aisimulate.sweeper.config import Workload
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ForwardPassEstimatorSpec, ReplaySpec
from aisimulate.sweeper.sample import unroll_sample

CONTROLS = {
    "enable_eplb": True,
    "wideep_num_slots": 256,
    "moe_backend": "deepep_moe",
    "attention_backend": "fa3",
    "gemm_quant_mode": "fp8",
    "moe_quant_mode": "fp8",
    "kvcache_quant_mode": "fp8",
    "fmha_quant_mode": "fp8",
    "comm_quant_mode": "fp8",
}


def _config():
    return {
        "engine": {
            "mode": "aggregated",
            "model": "deepseek-ai/DeepSeek-V3",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.14",
            "context_length": 4096,
            "nextn": 2,
            "nextn_accepted": 1.25,
            "enable_chunked_prefill": True,
            **CONTROLS,
            "workers": {"aggregated": {"kv_cache": {"capacity": {"type": "fixed", "blocks": 128}}}},
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2, "cached_prefix_tokens": 3},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 2},
        },
    }


def test_prediction_preserves_controls_in_canonical_timing_and_runtime():
    config = CorePredictionConfig.model_validate(_config())
    reloaded = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(reloaded)
    rank = spec.backend_deployment.agg_engine_args
    identity = rank["timing_model"]["config"]
    assert {name: identity[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == CONTROLS
    assert identity["worker_type"] == "aggregated"
    assert identity["nextn"] == rank["aic_nextn"] == 2
    assert identity["estimation_mode"] == "auto"
    assert identity["fallback_policy"] == "deny"
    assert rank["aic_nextn_accepted"] == 1.25
    assert rank["enable_chunked_prefill"] is True
    assert "aic_gemm_dtype" not in rank
    assert spec.workload["cached_prefix_tokens"] == 3
    metadata = spec.backend_deployment.performance_model_metadata["aggregated"]["config"]
    assert {name: metadata[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == CONTROLS
    assert metadata["nextn"] == 2


def test_recommendation_candidate_round_trip_keeps_request_and_identity():
    source = CoreRecommendationConfig.model_validate({**_config(), "optimization": {}})
    smart = recommendation_to_sweeper(source)
    space = smart.search_space
    sample = unroll_sample(
        search_space=space,
        selection={
            "deployment_mode": "agg",
            "backend": "sglang",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), replicas=1),
    )
    request = ForwardPassEstimatorResolver(space)._request(sample, "agg")
    identity = request.to_dict()
    assert {name: identity[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == CONTROLS
    assert smart.workload.cached_prefix_tokens == 3
    assert sample["nextn_accepted"] == 1.25
    with pytest.raises(ValueError, match="require a resolved canonical"):
        build_backend_deployment(sample, backend_version="0.5.14")
    resolved = ForwardPassEstimatorSpec(config=identity)
    deployment = build_backend_deployment(sample, backend_version="0.5.14", forward_pass_estimators={"agg": resolved})
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(mode="json"), goal={})
    predicted = _candidate_prediction(source, sample, spec, adapter_sections={})
    roundtrip = CorePredictionConfig.model_validate(predicted)
    assert {name: getattr(roundtrip.engine, name) for name in ENGINE_MODEL_CONTROL_FIELDS} == CONTROLS
    assert roundtrip.engine.nextn == 2
    assert roundtrip.engine.nextn_accepted == 1.25
    assert roundtrip.traffic.source.cached_prefix_tokens == 3


@pytest.mark.parametrize(
    "field,value",
    [("nextn_accepted", None), ("nextn_accepted", 3), ("nextn_accepted", float("nan")), ("wideep_num_slots", 0)],
)
def test_rejects_invalid_engine_request_controls(field, value):
    raw = _config()
    raw["engine"][field] = value
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("field", ["enable_eplb", "moe_backend", "gemm_quant_mode", "nextn"])
def test_rejects_controls_on_custom_timing(field):
    raw = _config()
    raw["engine"] = {
        key: value
        for key, value in raw["engine"].items()
        if key not in CONTROLS and key not in {"nextn", "nextn_accepted"}
    }
    raw["engine"][field] = 2 if field == "nextn" else CONTROLS[field]
    if field == "nextn":
        raw["engine"]["nextn_accepted"] = 1.25
    raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    with pytest.raises(ValidationError, match="require default timing"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("value", [-1, True, 1.5, 9])
def test_exact_prefix_rejects_invalid_lengths(value):
    with pytest.raises(ValidationError):
        Workload(isl=8, osl=2, concurrency=1, request_count=2, cached_prefix_tokens=value)


def test_exact_prefix_must_fit_shortest_random_input():
    with pytest.raises(ValidationError, match="shortest synthetic input"):
        Workload(isl=8, osl=2, concurrency=1, request_count=2, random_range_ratio=0.5, cached_prefix_tokens=5)


def test_explicit_accepted_count_reaches_expected_progress():
    # A guaranteed first draft plus a 25% second draft has mean 1.25.
    assert _accept_rates_for_expected(2, 1.25, role="aggregated") == "1,0.25"
    assert _accept_rates_for_expected(2, 0, role="aggregated") == "0,0"
    assert _accept_rates_for_expected(2, 2, role="aggregated") == "1,1"


def test_kv_relative_capacity_cache_separates_model_controls(monkeypatch):
    import aisimulate.sweeper.kv_load as kv

    calls = []

    def estimate(shape, **kwargs):
        calls.append(kwargs)
        return 128 if kwargs["model_controls"]["kvcache_quant_mode"] == "fp8" else 64

    monkeypatch.setattr(kv, "estimate_kv_tokens", estimate)
    kv._per_rank_capacity_tokens.cache_clear()
    common = dict(
        model_name="model",
        hardware_sku="gpu",
        backend="vllm",
        backend_version="v",
        systems_paths=(),
        max_num_tokens=8,
        max_batch_size=1,
        memory_fraction=0.9,
        nextn=0,
    )
    shape = ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1)
    try:
        assert kv._per_rank_capacity_tokens(shape, **common, model_controls=(("kvcache_quant_mode", "fp8"),)) == 128
        assert kv._per_rank_capacity_tokens(shape, **common, model_controls=(("kvcache_quant_mode", "bfloat16"),)) == 64
        assert len(calls) == 2
    finally:
        kv._per_rank_capacity_tokens.cache_clear()


@pytest.mark.parametrize(
    "overrides",
    [
        {"wideep_num_slots": 0},
        {"moe_backend": "unknown"},
        {"moe_backend": "deepep_moe"},
        {"estimation_mode": "fpm_interpolation", "enable_eplb": True},
    ],
)
def test_core_rejects_invalid_identity_before_data_fallback(overrides):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    with pytest.raises((ValueError, RuntimeError), match="(wideep_num_slots|moe_backend|EPLB)"):
        RustForwardPassPerfModel.best_available(
            {"model": "model", "system": "gpu", "backend": "vllm", "worker_type": "aggregated", **overrides}
        )


def test_migration_example_parses_and_preserves_shared_prefix():
    from pathlib import Path

    import yaml

    text = (Path(__file__).parents[2] / "docs/cli/migrate-from-aiconfigurator.md").read_text()
    example = (
        text.split("#### 4.6.3 Preserve pinned engine and request controls", 1)[1]
        .split("```yaml", 1)[1]
        .split("```", 1)[0]
    )
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(yaml.safe_load(example)))
    assert smart.workload.cached_prefix_tokens == 256
    assert smart.search_space.kvcache_quant_mode == "fp8"
    assert smart.search_space.agg_gpu_memory_utilization == 0.85


def test_kv_transfer_bytes_honor_explicit_quantization(tmp_path):
    import json

    from aisimulate.capacity import estimate_kv_bytes_per_token

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 64,
                "vocab_size": 32,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32,
            }
        )
    )
    # K+V, two KV heads per TP rank, width eight, two layers: 64 elements/token.
    kwargs = {"tp_size": 2, "pp_size": 1}
    assert estimate_kv_bytes_per_token(str(tmp_path), **kwargs, kvcache_quant_mode="bfloat16") == 128
    assert estimate_kv_bytes_per_token(str(tmp_path), **kwargs, kvcache_quant_mode="fp8") == 64


def test_memory_preflight_preserves_same_model_controls(monkeypatch):
    import aisimulate.sweeper.kv_estimate as memory

    seen = {}

    def estimate(*args, **kwargs):
        seen.update(kwargs)
        return {"total_kv_size_tokens": 128}

    monkeypatch.setattr(memory, "estimate_kv_cache", estimate)
    shape = ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8)
    assert (
        memory.estimate_kv_tokens(
            shape,
            model_name="model",
            hardware_sku="gpu",
            backend="sglang",
            backend_version="v",
            model_controls=CONTROLS,
        )
        == 128
    )
    assert {name: seen[name] for name in ENGINE_MODEL_CONTROL_FIELDS} == CONTROLS


def test_canonical_constructor_rejects_moe_controls_on_dense_model():
    from aisimulate_core.sdk import RustForwardPassPerfModel

    with pytest.raises((ValueError, RuntimeError), match="require an MoE model"):
        RustForwardPassPerfModel.best_available(
            {
                "model": "Qwen/Qwen3-32B",
                "system": "h200_sxm",
                "backend": "vllm",
                "worker_type": "aggregated",
                "estimation_mode": "op_level",
                "enable_eplb": True,
            }
        )


@pytest.mark.parametrize("slots", [0, -1, True, 1.5])
def test_capacity_rejects_invalid_slots_before_delegate(slots, monkeypatch):
    from aisimulate.capacity import estimate_num_gpu_blocks
    from aisimulate_core.sdk import memory

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", lambda **kwargs: pytest.fail("must reject before sizing"))
    with pytest.raises(ValueError, match="wideep_num_slots"):
        estimate_num_gpu_blocks(
            backend_name="sglang",
            system="h200_sxm",
            model_path="dense",
            tp_size=1,
            block_size=16,
            max_num_batched_tokens=128,
            wideep_num_slots=slots,
        )


@pytest.mark.parametrize("controls", [{"enable_eplb": True}, {"wideep_num_slots": 128}, {"moe_backend": "deepep_moe"}])
@pytest.mark.parametrize("is_moe", [False, True])
def test_memory_cannot_drop_moe_controls_in_naive_fallback(monkeypatch, controls, is_moe):
    from aisimulate_core.sdk import memory, models

    monkeypatch.setattr(models, "check_is_moe", lambda _: is_moe)

    def unsupported(*args, **kwargs):
        raise ValueError("native fixture unavailable")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", unsupported)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", lambda *a, **k: pytest.fail("controls lost"))
    with pytest.raises(ValueError, match="native fixture unavailable" if is_moe else "require an MoE model"):
        memory.estimate_kv_cache(
            "model",
            "h200_sxm",
            "sglang",
            max_num_tokens=128,
            max_batch_size=8,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
            **controls,
        )


@pytest.mark.parametrize("alias", [None, "auto", "none", "NULL", "  ", " auto "])
def test_kv_quantization_auto_aliases_preserve_model_dtype(monkeypatch, alias):
    from types import SimpleNamespace

    from aisimulate.capacity import estimate_kv_bytes_per_token
    from aisimulate_core.sdk.memory import NaiveKVCacheEstimator

    estimator = SimpleNamespace(dtype_bytes=2)
    estimator.kv_bytes_per_token = lambda: 100 * estimator.dtype_bytes
    monkeypatch.setattr(NaiveKVCacheEstimator, "from_model_path", lambda *a, **k: estimator)
    assert estimate_kv_bytes_per_token("model", tp_size=1, pp_size=1, kvcache_quant_mode=alias) == 200


def test_optional_runner_must_advertise_new_request_capabilities():
    from dataclasses import replace

    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import RunnerCapabilities

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_config()))
    legacy = RunnerCapabilities(supported_backend_topologies=(("*", "*"),))
    with pytest.raises(ValueError, match="cached_prefix_tokens"):
        legacy.require_compatible(spec)
    without_prefix = replace(spec, workload={**spec.workload, "cached_prefix_tokens": 0})
    with pytest.raises(ValueError, match="engine model controls"):
        legacy.require_compatible(without_prefix)
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)


def test_recommendation_rejects_unconfigured_backend_version():
    raw = {**_config(), "optimization": {}}
    raw["engine"]["backend_version"] = {"vllm": "0.24.0"}
    with pytest.raises(ValidationError, match="unconfigured backend"):
        CoreRecommendationConfig.model_validate(raw)


@pytest.mark.parametrize("mode", ["afd", "afd+pd"])
def test_cached_prefix_rejected_by_direct_sweeper_and_replay(mode):
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.replay import BackendDeploymentSpec

    with pytest.raises(ValueError, match="cached_prefix_tokens is unsupported for AFD"):
        SmartSearchConfig(
            search_space={
                "model_name": "model",
                "hardware_sku": "h200_sxm",
                "deployment_mode": [mode],
                "afd_phase": "decode",
                "afd_batch_size_candidates": [8],
            },
            workload={"isl": 8, "osl": 2, "request_count": 2, "concurrency": 1, "cached_prefix_tokens": 4},
        )
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(deployment_mode=mode, backend="vllm", backend_version="v"),
        workload={"cached_prefix_tokens": 4},
        goal={},
    )
    with pytest.raises(ValueError, match="cached_prefix_tokens is unsupported for AFD"):
        EngineReplayRunnerFactory().create(0).run(spec)


@pytest.mark.parametrize("moe_backend", [None, "default"])
def test_default_prediction_payload_remains_compatible_with_older_runners(moe_backend):
    from aisimulate.sweeper.replay import RunnerCapabilities

    raw = _config()
    for name in (*CONTROLS, "nextn", "nextn_accepted"):
        raw["engine"].pop(name, None)
    raw["engine"]["moe_backend"] = moe_backend
    raw["traffic"]["source"]["cached_prefix_tokens"] = 0
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    identity = spec.backend_deployment.agg_engine_args["timing_model"]["config"]
    assert not {"moe_backend", "enable_eplb", "wideep_num_slots"} & identity.keys()
    RunnerCapabilities(supported_backend_topologies=(("*", "*"),)).require_compatible(spec)


def test_memory_cannot_drop_attention_backend_in_naive_fallback(monkeypatch):
    from aisimulate_core.sdk import memory

    def unsupported(*args, **kwargs):
        raise ValueError("native fixture unavailable")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", unsupported)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", lambda *a, **k: pytest.fail("control lost"))
    with pytest.raises(ValueError, match="native fixture unavailable"):
        memory.estimate_kv_cache(
            "model",
            "h200_sxm",
            "vllm",
            max_num_tokens=128,
            max_batch_size=8,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
            attention_backend="fa3",
        )


@pytest.mark.parametrize(
    "name,inactive,active",
    [
        ("enable_eplb", False, True),
        ("wideep_num_slots", None, 128),
        ("moe_backend", "default", "deepep_moe"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "aic_"])
def test_inactive_nested_control_cannot_mask_active_flat_control(name, inactive, active, prefix):
    from aisimulate.sweeper.replay import BackendDeploymentSpec, RunnerCapabilities

    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args={prefix + name: active, "timing_model": {"config": {name: inactive}}},
        ),
        workload={},
        goal={},
    )
    with pytest.raises(ValueError, match="engine model controls"):
        RunnerCapabilities(supported_backend_topologies=(("*", "*"),)).require_compatible(spec)


def test_explicit_default_moe_backend_is_inactive_with_custom_timing():
    from aisimulate.sweeper.config import SearchSpace

    raw = _config()
    for name in (*CONTROLS, "nextn", "nextn_accepted"):
        raw["engine"].pop(name, None)
    raw["engine"]["moe_backend"] = "default"
    raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert spec.backend_deployment.agg_engine_args["timing_model"]["type"] == "fixed"
    space = SearchSpace(
        model_name="model",
        hardware_sku="gpu",
        moe_backend="default",
        agg_timing_model={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
    )
    sample = unroll_sample(
        search_space=space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    sample["agg_num_gpu_blocks"] = 128
    deployment = build_backend_deployment(sample, backend_version="test")
    assert deployment.agg_engine_args["timing_model"]["type"] == "fixed"
