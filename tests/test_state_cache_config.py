# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual G1 state-cache geometry reaches replay without AIC estimation."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from aisimulate import capacity, compiler, runner
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.engine import KvCachePredictionConfig, StateCacheConfig
from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_role

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

STATE = {"bytes_per_request": 1500}
KV = {"block_size": 64, "bytes_per_token": 16, "capacity": {"type": "fixed", "bytes": 8192}, "state_cache": STATE}
RANK = {"block_size": 64, "num_gpu_blocks": 8, "kv_cache_bytes_per_token": 16, "state_cache": STATE}


def _public(*, timing="fixed", **cache_overrides):
    return {
        "engine": {
            "mode": "aggregated",
            "backend": "vllm",
            "model": "manual-state-smoke",
            "hardware": "h200_sxm",
            "context_length": 2048,
            "workers": {
                "aggregated": {
                    "kv_cache": {**KV, **cache_overrides},
                    "timing": (
                        {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                        if timing == "fixed"
                        else {"type": timing}
                    ),
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


class RecordingRuntime:
    execution_spec = None

    def run_replay_json(self, payload):
        self.execution_spec = json.loads(payload)
        return json.dumps({"duration_ms": 1.0, "completed_requests": 1})


@pytest.fixture
def forbid_estimators(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("manual state_cache must not invoke a KV estimator")

    for module, names in (
        (capacity, ("estimate_num_gpu_blocks", "estimate_kv_bytes_per_token", "materialize_aic_num_gpu_blocks")),
        (compiler, ("estimate_kv_bytes_per_token", "materialize_aic_num_gpu_blocks")),
        (runner, ("materialize_aic_num_gpu_blocks",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, unexpected)


@pytest.mark.parametrize("timing", ["fixed", "polynomial", "default"])
def test_manual_geometry_reaches_native_wire_without_estimation(forbid_estimators, timing):
    config = CorePredictionConfig.model_validate(_public(timing=timing))
    # Public export/reload must not make unused defaults conflict with the manual geometry.
    config = CorePredictionConfig.model_validate(config.model_dump(mode="json"))
    spec = prediction_to_replay_spec(config)
    args = spec.backend_deployment.agg_engine_args
    assert args["state_cache"] == STATE
    assert not {"gpu_memory_utilization", "cuda_graph_reserved_bytes"} & args.keys()
    assert args["num_gpu_blocks"] == 8
    assert args["block_size"] == 64
    assert args["kv_cache_bytes_per_token"] == 16
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    engine = runtime.execution_spec["spec"]["engine"]
    assert engine["num_gpu_blocks_is_explicit"] is True
    assert engine["rank"]["state_cache"] == STATE
    assert engine["rank"]["block_size"] == 64
    assert engine["rank"]["num_gpu_blocks"] == 8
    if timing == "default":
        assert engine["rank"]["timing_model"]["config"]["kv_block_size"] == 64


@pytest.mark.parametrize("field", STATE)
def test_manual_geometry_requires_all_fields(field):
    state = {key: value for key, value in STATE.items() if key != field}
    with pytest.raises(ValidationError, match=field):
        StateCacheConfig.model_validate(state)


@pytest.mark.parametrize("field", STATE)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16", 1 << 64])
def test_manual_geometry_rejects_invalid_unsigned_integers(field, value):
    with pytest.raises(ValidationError, match=field):
        StateCacheConfig.model_validate({**STATE, field: value})


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"block_size": 1}, "at least two"),
        ({"bytes_per_token": (1 << 64) - 1}, "overflows"),
        ({"capacity": {"type": "fixed", "bytes": 2048}}, "must fit"),
        ({"capacity": {"type": "fixed", "bytes": 8192, "blocks": 8}}, "only one"),
        ({"capacity": {"type": "fixed"}}, "blocks or bytes"),
        ({"capacity": {"type": "default", "bytes": 8192}}, "rejects"),
        ({"capacity": {"type": "fixed", "bytes": 8192, "memory_fraction": 0.9}}, "rejects"),
        ({"block_size": None}, "explicit positive block_size"),
        ({"bytes_per_token": "auto"}, "explicit positive bytes_per_token"),
        ({"capacity": {"type": "default"}}, "requires fixed"),
        ({"host_offload": {"num_host_blocks": 8}}, "G1 only"),
        (
            {"host_offload": {"num_host_blocks": 8}, "g3_offload": {"scope": "worker_local", "num_g3_blocks": 8}},
            "G1 only",
        ),
        ({"state_cache": {"bytes_per_request": 1500, "tokens_per_block": 64}}, "Extra inputs"),
    ],
)
def test_manual_geometry_rejects_unusable_capacity(overrides, error):
    with pytest.raises(ValidationError, match=error):
        KvCachePredictionConfig.model_validate({**KV, **overrides})


@pytest.mark.parametrize(
    "capacity,expected",
    [
        ({"type": "fixed", "bytes": 8192}, 8),
        ({"type": "fixed", "bytes": 8193}, 8),
        ({"type": "fixed", "blocks": 8}, 8),
    ],
)
def test_fixed_bytes_and_existing_blocks_compile_to_same_pool(forbid_estimators, capacity, expected):
    config = CorePredictionConfig.model_validate(_public(capacity=capacity))
    args = prediction_to_replay_spec(config).backend_deployment.agg_engine_args
    assert args["num_gpu_blocks"] == expected
    assert args["state_cache"] == {"bytes_per_request": 1500}


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8192", 1 << 64])
def test_capacity_bytes_requires_positive_u64(value):
    with pytest.raises(ValidationError):
        KvCachePredictionConfig.model_validate({**KV, "capacity": {"type": "fixed", "bytes": value}})


def test_byte_capacity_is_not_owned_by_state_cache():
    config = CorePredictionConfig.model_validate(_public(state_cache=None))
    args = prediction_to_replay_spec(config).backend_deployment.agg_engine_args
    assert args["num_gpu_blocks"] == 8
    assert "state_cache" not in args


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_public_manual_cache_rejects_other_backends(backend):
    raw = _public()
    raw["engine"]["backend"] = backend
    with pytest.raises(ValidationError, match="state_cache requires backend=vllm"):
        CorePredictionConfig.model_validate(raw)


def test_public_manual_cache_rejects_pd():
    raw = _public()
    worker = raw["engine"]["workers"]["aggregated"]
    raw["engine"]["mode"] = "disaggregated"
    raw["engine"]["workers"] = {"prefill": worker, "decode": {}}
    with pytest.raises(ValidationError, match="state_cache requires"):
        CorePredictionConfig.model_validate(raw)


def test_recommendation_rejects_manual_cache_instead_of_dropping_it():
    raw = {"engine": _public()["engine"], "optimization": {"target": "throughput"}}
    with pytest.raises(ValidationError, match="state_cache currently supports prediction only"):
        CoreRecommendationConfig.model_validate(raw)


@pytest.mark.parametrize("defaults", [{}, {"cuda_graph_reserved_bytes": 0, "kv_transfer_timing_mode": "full_prompt"}])
@pytest.mark.parametrize("nested", [False, True])
def test_direct_engine_manual_geometry_does_not_estimate_or_replace_shared_fields(forbid_estimators, nested, defaults):
    raw = {**RANK, **defaults, "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}
    engine = _materialize_engine_role("vllm", "", {}, {"rank": raw} if nested else raw, "aggregated")
    assert engine["rank"]["state_cache"] == STATE
    assert engine["rank"]["block_size"] == 64
    assert engine["rank"]["num_gpu_blocks"] == 8
    assert engine["num_gpu_blocks_is_explicit"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "override",
    [
        {"num_gpu_blocks": 0},
        {"block_size": None},
        {"kv_cache_bytes_per_token": None},
        {"gpu_memory_utilization": 0.9},
        {"cuda_graph_reserved_bytes": 1},
        {"native_host_offload": {"num_host_blocks": 8}},
        {"g3_offload": {"scope": "worker_local", "num_g3_blocks": 8}},
        {"kv_transfer_bytes_per_token": 16},
        {"kv_transfer_bandwidth": 0.0},
        {"kv_transfer_timing_mode": "destination_missing"},
    ],
)
def test_direct_engine_rejects_unsupported_overrides_before_estimating(forbid_estimators, nested, override):
    raw = {**RANK, **override}
    with pytest.raises(ValueError, match="state_cache"):
        _materialize_engine_role("vllm", "", {}, {"rank": raw} if nested else raw, "aggregated")


@pytest.mark.parametrize("backend,role", [("sglang", "aggregated"), ("vllm", "prefill"), ("vllm", "decode")])
def test_direct_engine_rejects_unsupported_topologies_before_estimating(forbid_estimators, backend, role):
    with pytest.raises(ValueError, match="state_cache requires"):
        _materialize_engine_role(backend, "", {}, {"state_cache": STATE}, role)


@pytest.mark.parametrize("field", ["num_gpu_blocks", "block_size", "kv_cache_bytes_per_token"])
def test_native_manual_state_requires_explicit_shared_fields(forbid_estimators, field):
    raw = {key: value for key, value in RANK.items() if key != field}
    with pytest.raises(ValueError, match=field):
        _materialize_engine_role("vllm", "", {}, raw, "aggregated")


def test_manual_state_reports_unavailable_memory_without_estimating(forbid_estimators):
    diagnostics = {}
    raw = {**RANK, "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}
    engine = _materialize_engine_role("vllm", "", {}, raw, "aggregated", memory_diagnostics=diagnostics)
    assert engine["rank"]["num_gpu_blocks"] == 8
    assert diagnostics["aggregated"]["status"] == "unavailable"
    assert "total_gpu_capacity_bytes" not in diagnostics["aggregated"]


@pytest.mark.parametrize("role", ["agg_engine_args", "prefill_engine_args", "decode_engine_args"])
@pytest.mark.parametrize("nested", [False, True])
def test_state_cache_requires_runner_capability(role, nested):
    from aisimulate.sweeper.replay import RunnerCapabilities

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public()))
    args = {"rank": dict(RANK)} if nested else dict(RANK)
    deployment = replace(
        spec.backend_deployment, agg_engine_args=None, prefill_engine_args=None, decode_engine_args=None
    )
    spec = replace(spec, backend_deployment=replace(deployment, **{role: args}))
    capabilities = RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))
    with pytest.raises(ValueError, match="runner does not support state_cache"):
        capabilities.require_compatible(spec)
    replace(capabilities, supports_state_cache=True).require_compatible(spec)
    rank = args["rank"] if nested else args
    rank["state_cache"] = None
    capabilities.require_compatible(spec)
    del rank["state_cache"]
    capabilities.require_compatible(spec)


def test_native_engine_advertises_state_cache_support():
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_public()))
    capabilities = EngineReplayRunnerFactory().capabilities()
    assert capabilities.supports_state_cache
    capabilities.require_compatible(spec)


def test_cli_rejects_state_cache_before_creating_unsupported_runner(tmp_path, monkeypatch, capsys):
    import yaml

    import aisimulate.main as cli
    from aisimulate.sweeper.replay import RunnerCapabilities

    class UnsupportedFactory:
        def capabilities(self):
            return RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))

        def create(self, worker_id):
            pytest.fail("unsupported runner must not be created")

    config = tmp_path / "state.yaml"
    config.write_text(yaml.safe_dump(_public()), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda stack: UnsupportedFactory())
    with pytest.raises(SystemExit) as error:
        cli.main(["predict", "--stack", "dynamo", "--config", str(config), "--output-dir", str(output)])
    assert error.value.code == 2
    assert "runner does not support state_cache" in capsys.readouterr().err
    assert not output.exists()


def test_aligned_state_config_survives_public_export_and_engine_handoff(forbid_estimators):
    state = STATE
    config = CorePredictionConfig.model_validate(_public(state_cache=state, prefix_match_unit=16))
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(config)
    assert spec.backend_deployment.agg_engine_args["state_cache"] == state
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec["spec"]["engine"]["rank"]["state_cache"] == state
    assert runtime.execution_spec["spec"]["engine"]["rank"]["prefix_match_unit"] == 16


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1536", 1 << 64, 63, 65])
def test_aligned_state_config_rejects_invalid_geometry(value):
    with pytest.raises(ValidationError, match="prefix_match_unit"):
        CorePredictionConfig.model_validate(_public(prefix_match_unit=value))


def test_aligned_state_config_rejects_speculation():
    config = _public(prefix_match_unit=16)
    config["engine"]["speculation"] = {
        "kind": "ngram",
        "num_speculative_tokens": 1,
        "acceptance_rates": [1.0],
        "seed": 42,
    }
    with pytest.raises(ValidationError, match="speculative"):
        CorePredictionConfig.model_validate(config)
    with pytest.raises(ValueError, match="speculative"):
        _materialize_engine_role("vllm", "", {}, {**RANK, "prefix_match_unit": 16, "aic_nextn": 1}, "agg")


@pytest.mark.parametrize("nextn", [1, 5])
def test_aligned_state_config_rejects_positive_nextn_at_public_validation(nextn):
    config = _public(timing="default", prefix_match_unit=16)
    config["engine"].update(nextn=nextn, nextn_accepted=float(nextn))
    with pytest.raises(ValidationError, match="prefix_match_unit.*nextn > 0"):
        CorePredictionConfig.model_validate(config)


def test_aligned_state_config_allows_zero_nextn_through_engine_handoff(forbid_estimators):
    raw = _public(timing="default", prefix_match_unit=16)
    raw["engine"]["nextn"] = 0
    config = CorePredictionConfig.model_validate(raw)
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(config)
    assert "aic_nextn" not in spec.backend_deployment.agg_engine_args
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    assert rank.get("aic_nextn") is None
    assert rank["prefix_match_unit"] == 16


@pytest.mark.parametrize(
    "shared_prefix_tokens, expected_reused_tokens",
    [
        pytest.param(24192, 24192, id="partial-checkpoint-hit"),
        pytest.param(24191, 23040, id="one-token-before-checkpoint"),
        pytest.param(0, 0, id="no-shared-prefix"),
    ],
)
def test_aligned_state_config_reuses_retained_native_checkpoints(
    forbid_estimators, shared_prefix_tokens, expected_reused_tokens
):
    raw = _public(
        block_size=1536,
        prefix_match_unit=128,
        capacity={"type": "fixed", "blocks": 512},
        state_cache={"bytes_per_request": 1536 * 16},
    )
    raw["engine"]["context_length"] = 32768
    raw["engine"]["workers"]["aggregated"]["scheduler"] = {"max_batched_tokens": 8192}
    raw["traffic"]["source"].update(input_tokens=24300, output_tokens=2, cached_prefix_tokens=shared_prefix_tokens)
    raw["traffic"]["stop"]["requests"] = 2
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    result = EngineReplayRunnerFactory().create(0).run(spec)
    assert result.metrics["completed_requests"] == 2
    assert result.metrics["total_input_tokens"] == 48600
    assert result.metrics["total_output_tokens"] == 4
    # The first request is cold. Only the second can reuse a retained checkpoint;
    # a positive hit ratio alone would also accept an incorrect full-block hit.
    assert result.metrics["committed_prefill_tokens"] == 48600 - expected_reused_tokens
    assert result.metrics["prefix_cache_reused_ratio"] == pytest.approx(expected_reused_tokens / 48600)


@pytest.fixture
def inferred_prefix_payload():
    model, tp, token_bytes, state_bytes = "moonshotai/Kimi-K3", 8, 13824, 61046784
    payload = _public(
        block_size=1536,
        bytes_per_token=token_bytes,
        prefix_match_unit=128,
        capacity={"type": "fixed", "blocks": 512},
        state_cache={
            "layout": "auto",
            "model_dtype": "auto",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "float32",
        },
    )
    payload["engine"].update(model=model, context_length=32768)
    payload["engine"]["workers"]["aggregated"].update(
        parallelism={"tensor": tp}, scheduler={"max_batched_tokens": 8192}
    )
    payload["traffic"]["source"].update(input_tokens=24300, output_tokens=2)
    payload["traffic"]["stop"]["requests"] = 2
    return payload, state_bytes


@pytest.mark.parametrize(
    "prefix_unit, shared_prefix, expected_reused",
    [
        pytest.param(128, 24192, 24192, id="partial-hit"),
        pytest.param(128, 24191, 23040, id="before-partial-checkpoint"),
        pytest.param(128, 0, 0, id="miss"),
        pytest.param(1536, 24192, 23040, id="full-block-unit"),
        # Without the alignment option, the 8192-token scheduler chunks do not
        # end on physical pages, so no reusable state was actually computed.
        pytest.param(None, 24192, 0, id="legacy-unaligned-chunks"),
    ],
)
def test_inferred_state_matches_manual_partial_prefix_replay(
    forbid_estimators, inferred_prefix_payload, prefix_unit, shared_prefix, expected_reused
):
    payload, state_bytes = inferred_prefix_payload
    cache = payload["engine"]["workers"]["aggregated"]["kv_cache"]
    cache["prefix_match_unit"] = prefix_unit
    payload["traffic"]["source"]["cached_prefix_tokens"] = shared_prefix
    config = CorePredictionConfig.model_validate(payload)
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    inferred = prediction_to_replay_spec(config)
    info = inferred.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert info["source"] == "inferred"
    assert info["bytes_per_request"] == state_bytes
    assert info["state_blocks"] == 3
    assert info["allocated_bytes_per_request"] == 3 * 1536 * cache["bytes_per_token"]
    assert info["ssm_dtype"] == "float32"
    args = inferred.backend_deployment.agg_engine_args
    # Only the resolved byte size crosses the native boundary, never SDK knobs.
    assert args["state_cache"] == {"bytes_per_request": state_bytes}
    assert args.get("prefix_match_unit") == prefix_unit
    cache["state_cache"] = {"bytes_per_request": state_bytes}
    manual = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    assert manual.backend_deployment.agg_engine_args == args
    observed = []
    for spec in (inferred, manual):
        metrics = EngineReplayRunnerFactory().create(0).run(spec).metrics
        assert metrics["completed_requests"] == 2
        assert metrics["total_input_tokens"] == 48600
        assert metrics["total_output_tokens"] == 4
        assert metrics["committed_prefill_tokens"] == 48600 - expected_reused
        assert metrics["prefix_cache_reused_ratio"] == pytest.approx(expected_reused / 48600)
        observed.append((metrics["committed_prefill_tokens"], metrics["prefix_cache_reused_ratio"]))
    assert observed[0] == observed[1]


def test_prefix_match_unit_changes_neither_inferred_geometry_nor_dtype(forbid_estimators, inferred_prefix_payload):
    payload, state_bytes = inferred_prefix_payload
    cache = payload["engine"]["workers"]["aggregated"]["kv_cache"]
    by_dtype = {}
    for dtype in ("auto", "float32"):
        cache["state_cache"]["mamba_cache_dtype"] = dtype
        sizes = []
        for unit in (None, 128, 1536):
            cache["prefix_match_unit"] = unit
            spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
            sizes.append(spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"])
        assert sizes[0] == sizes[1] == sizes[2]
        assert sizes[0]["bytes_per_request"] == state_bytes
        assert sizes[0]["state_blocks"] == 3
        by_dtype[dtype] = sizes[0]
    # Both dtypes fit the supplied physical page: FP32 uses more of the page
    # without growing it. Prefix hash granularity must not enter either formula.
    assert by_dtype["auto"]["conv_dtype"] == "bfloat16"
    assert by_dtype["float32"]["conv_dtype"] == "float32"
    assert by_dtype["float32"]["raw_bytes_per_request"] > by_dtype["auto"]["raw_bytes_per_request"]
    assert by_dtype["float32"]["padding_bytes_per_request"] < by_dtype["auto"]["padding_bytes_per_request"]


def test_prefix_match_unit_requires_state_and_old_alignment_field_is_rejected():
    with pytest.raises(ValidationError, match="requires state_cache"):
        CorePredictionConfig.model_validate(_public(state_cache=None, prefix_match_unit=16))
    with pytest.raises(ValidationError, match="Extra inputs"):
        CorePredictionConfig.model_validate(_public(state_cache={**STATE, "prefill_block_size": 1536}))


@pytest.fixture
def kda_model(tmp_path):
    geometry = {
        "model_type": "kimi_linear",
        "num_hidden_layers": 4,
        "dtype": "bfloat16",
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(geometry))
    return tmp_path, geometry


def _auto_public(model, **sizing):
    payload = _public(state_cache=sizing)
    payload["engine"]["model"] = str(model)
    return payload


def test_inferred_size_roundtrip_and_native_wire(kda_model, forbid_estimators):
    path, _ = kda_model
    config = CorePredictionConfig.model_validate(_auto_public(path))
    config = CorePredictionConfig.model_validate(config.model_dump(mode="json"))
    assert config.engine.workers.aggregated.kv_cache.state_cache.bytes_per_request is None
    spec = prediction_to_replay_spec(config)
    info = spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    # Three conv windows: 3*3*2*8*2=288; FP32 matrix: 2*8*8*4=512.
    assert info["raw_bytes_per_layer"] == 800
    assert info["padded_bytes_per_layer"] == 1024
    assert info["bytes_per_request"] == 3072
    assert info["source"] == "inferred"
    assert info["state_blocks"] == 3
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec["spec"]["engine"]["rank"]["state_cache"] == {"bytes_per_request": 3072}


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"model_type": "mamba2"}, "unknown model"),
        ({"dtype": "float32"}, "set model_dtype"),
    ],
)
def test_inference_fails_closed(kda_model, change, reason):
    path, geometry = kda_model
    (path / "config.json").write_text(json.dumps({**geometry, **change}))
    config = CorePredictionConfig.model_validate(_auto_public(path))
    with pytest.raises(ValueError, match=reason):
        prediction_to_replay_spec(config)


@pytest.mark.parametrize("parallel,reason", [({"pipeline": 2}, "PP=1"), ({"tensor": 3}, "divisible")])
def test_inference_rejects_unsupported_parallelism(kda_model, parallel, reason):
    payload = _auto_public(kda_model[0])
    payload["engine"]["workers"]["aggregated"]["parallelism"] = parallel
    with pytest.raises(ValueError, match=reason):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))


def test_unknown_layout_override_and_disabled(kda_model):
    payload = _auto_public(kda_model[0], layout="future-layout")
    with pytest.raises(ValueError, match="unknown layout"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    payload["engine"]["model"] = "must-not-load"
    payload["engine"]["workers"]["aggregated"]["kv_cache"]["state_cache"]["bytes_per_request"] = 1500
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    assert spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]["source"] == "overridden"
    payload["engine"]["workers"]["aggregated"]["kv_cache"]["state_cache"] = None
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    assert spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]["source"] == "disabled"
    assert "state_cache" not in spec.backend_deployment.agg_engine_args


def test_inferred_capacity_validation(kda_model):
    payload = _auto_public(kda_model[0])
    payload["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"] = {
        "type": "fixed",
        "blocks": 3,
    }
    with pytest.raises(ValueError, match="capacity must fit"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))


def test_fractional_attention_geometry_rejected(kda_model):
    path, geometry = kda_model
    geometry["num_hidden_layers"] = 6
    # Three attention layers cannot divide 16 bytes/token.
    geometry["linear_attn_config"]["full_attn_layers"] = [4, 5, 6]
    (path / "config.json").write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="divide evenly"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(_auto_public(path)))


def test_existing_typed_state_config_remains_accepted():
    cache = KvCachePredictionConfig(**{**KV, "state_cache": StateCacheConfig(**STATE)})
    assert cache.state_cache.bytes_per_request == 1500
    payload = _public()
    payload["engine"]["workers"]["aggregated"]["kv_cache"] = cache
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    assert spec.backend_deployment.agg_engine_args["state_cache"] == STATE
    assert spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]["source"] == "overridden"


@pytest.mark.parametrize("field", ["mamba_cache_dtype", "mamba_ssm_cache_dtype"])
def test_cache_dtype_accepts_only_pinned_vllm_cli_values(field):
    with pytest.raises(ValidationError, match=field):
        CorePredictionConfig.model_validate(_public(state_cache={field: "bfloat16"}))


@pytest.mark.parametrize("tp,conv_dtype,raw_bytes", [(1, "auto", 800), (2, "auto", 400), (1, "float32", 1088)])
def test_kda_single_working_state_geometry(kda_model, forbid_estimators, tp, conv_dtype, raw_bytes):
    path, _ = kda_model
    payload = _auto_public(path, mamba_cache_dtype=conv_dtype)
    worker = payload["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"tensor": tp}
    worker["kv_cache"].update(bytes_per_token=48, capacity={"type": "fixed", "blocks": 8})
    config = CorePredictionConfig.model_validate(payload)
    config = CorePredictionConfig.model_validate(config.model_dump(mode="json"))
    assert config.engine.workers.aggregated.kv_cache.state_cache.layout == "auto"
    spec = prediction_to_replay_spec(config)
    info = spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert info["layout"] == "vllm-kda-a474da28"
    assert info["raw_bytes_per_layer"] == raw_bytes  # Three conv windows + one FP32 matrix, not five slots.
    assert info["ssm_dtype"] == "float32"
    assert info["bytes_per_request"] == 9216
    assert spec.backend_deployment.agg_engine_args["state_cache"] == {"bytes_per_request": 9216}


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("kda_layers", [0, 1, 2], "1-based"),
        ("kda_layers", [1, 1, 2], "duplicate"),
        ("kda_layers", [True, 2, 3], "1-based"),
        ("full_attn_layers", [3, 4], "partition"),
        ("kda_layers", [1, 2], "partition"),
        ("full_attn_layers", [], "1-based"),
        ("num_heads", 0, "num_heads"),
        ("head_dim", None, "head_dim"),
        ("num_k_heads", 1, "asymmetric"),
    ],
)
def test_kda_rejects_unrecognized_geometry(kda_model, key, value, error):
    path, geometry = kda_model
    geometry["linear_attn_config"][key] = value
    (path / "config.json").write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match=error):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(_auto_public(path)))


def test_kda_rejects_undersized_attention_page(kda_model):
    payload = _auto_public(kda_model[0])
    payload["engine"]["workers"]["aggregated"]["kv_cache"]["bytes_per_token"] = 8
    with pytest.raises(ValueError, match="attention page is smaller"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))


def test_kda_requires_fixed_fp32_recurrent_dtype(kda_model):
    with pytest.raises(ValueError, match="always uses float32"):
        prediction_to_replay_spec(
            CorePredictionConfig.model_validate(_auto_public(kda_model[0], mamba_ssm_cache_dtype="float16"))
        )


def test_kda_rejects_unqualified_speculation(kda_model):
    payload = _auto_public(kda_model[0])
    payload["engine"]["speculation"] = {"kind": "ngram", "num_speculative_tokens": 2, "acceptance_rates": [0.5, 0.5]}
    with pytest.raises(ValueError, match="speculative KDA"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))


@pytest.mark.parametrize("tp,raw_bytes", [(1, 6512640), (2, 3256320), (8, 814080)])
def test_shipped_k3_nested_geometry(tp, raw_bytes, forbid_estimators):
    payload = _auto_public("moonshotai/Kimi-K3")
    worker = payload["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"tensor": tp}
    worker["kv_cache"].update(block_size=6144, bytes_per_token=27648, capacity={"type": "fixed", "blocks": 8})
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    info = spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert info["recurrent_layers_per_rank"] == 69
    assert info["raw_bytes_per_layer"] == raw_bytes
    assert info["bytes_per_request"] == 488374272
    assert info["state_blocks"] == 3
    assert info["allocated_bytes_per_request"] == 509607936  # 69/24 pages rounds up only after the per-rank sum.


def test_k3_explicit_effective_token_geometry_does_not_reshard_state():
    payload = _auto_public("moonshotai/Kimi-K3")
    worker = payload["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"tensor": 8}
    worker["kv_cache"].update(block_size=12288, bytes_per_token=1728, capacity={"type": "fixed", "blocks": 8})
    info = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(payload)
    ).backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert info["raw_bytes_per_layer"] == 814080
    assert info["padded_bytes_per_layer"] == 884736
    assert info["bytes_per_request"] == 61046784
    assert info["allocated_bytes_per_request"] == 63700992

    worker["kv_cache"].update(block_size=1536, bytes_per_token=13824)
    physical = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(payload)
    ).backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert physical == info  # Byte-geometry normalization, not a DCP execution qualification.


def test_kda_model_dtype_override_requires_explicit_cache_dtype(kda_model):
    with pytest.raises(ValueError, match="explicit mamba_cache_dtype"):
        prediction_to_replay_spec(
            CorePredictionConfig.model_validate(_auto_public(kda_model[0], model_dtype="float16"))
        )
    payload = _auto_public(kda_model[0], model_dtype="float16", mamba_cache_dtype="float16")
    payload["engine"]["workers"]["aggregated"]["kv_cache"]["bytes_per_token"] = 32
    info = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(payload)
    ).backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert info["conv_dtype"] == "float16"
    assert info["raw_bytes_per_layer"] == 800


def test_public_state_estimator_accepts_loaded_config_without_loader(kda_model, monkeypatch):
    from copy import deepcopy
    from types import MappingProxyType

    from aisimulate_core.sdk import estimate_state_cache, state_memory

    geometry = kda_model[1]
    before = deepcopy(geometry)
    monkeypatch.setattr(
        state_memory, "_load_state_config", lambda *a, **k: pytest.fail("loaded config must not reload")
    )
    result = estimate_state_cache(
        model_config=MappingProxyType(geometry), backend="vllm", block_size=64, kv_bytes_per_token=16
    )
    assert geometry == before
    assert result["bytes_per_request"] == 3072
    assert result["raw_bytes_per_request"] == 2400
    assert result["padding_bytes_per_request"] == 672
    assert result["backend"] == "vllm"
    assert result["backend_revision"] == result["vllm_revision"]
    assert "state_blocks" not in result and "allocated_bytes_per_request" not in result


def test_public_state_estimator_path_and_config_agree(kda_model):
    from aisimulate_core.sdk import estimate_state_cache

    path, geometry = kda_model
    controls = {"backend": "vllm", "block_size": 64, "kv_bytes_per_token": 48}
    assert estimate_state_cache(str(path), **controls) == estimate_state_cache(model_config=geometry, **controls)


def test_public_state_estimator_has_no_pool_capacity_dependency(forbid_estimators, monkeypatch):
    from aisimulate_core.sdk import estimate_state_cache, memory

    monkeypatch.setattr(
        memory, "estimate_kv_cache", lambda **k: pytest.fail("state size must not estimate pool capacity")
    )
    result = estimate_state_cache(
        "moonshotai/Kimi-K3", backend="vllm", tp_size=8, block_size=768, kv_bytes_per_token=27648
    )
    assert result["bytes_per_request"] == 61046784
    assert result["raw_bytes_per_request"] == 56171520
    assert result["padding_bytes_per_request"] == 4875264


def test_cli_passes_complete_state_controls_to_public_estimator(kda_model, monkeypatch):
    from aisimulate import state_size
    from aisimulate_core.sdk import estimate_state_cache

    calls = []

    def record(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return estimate_state_cache(model_path, **kwargs)

    monkeypatch.setattr(state_size, "estimate_state_cache", record)
    payload = _auto_public(
        kda_model[0],
        layout="vllm-kda-a474da28",
        model_dtype="bfloat16",
        mamba_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
    )
    worker = payload["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"tensor": 2}
    worker["kv_cache"]["bytes_per_token"] = 32
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    assert calls == [
        (
            str(kda_model[0]),
            {
                "backend": "vllm",
                "tp_size": 2,
                "pp_size": 1,
                "block_size": 64,
                "kv_bytes_per_token": 32,
                "model_dtype": "bfloat16",
                "mamba_cache_dtype": "float16",
                "mamba_ssm_cache_dtype": "float32",
                "num_speculative_tokens": 0,
                "layout": "vllm-kda-a474da28",
                "allow_hf_config_download": True,
            },
        )
    ]
    assert spec.backend_deployment.agg_engine_args["state_cache"] == {"bytes_per_request": 6144}


@pytest.mark.parametrize("sizing", [None, {"bytes_per_request": 1500}])
def test_cli_override_and_disabled_skip_public_estimator(sizing, monkeypatch):
    from aisimulate import state_size

    monkeypatch.setattr(state_size, "estimate_state_cache", lambda *a, **k: pytest.fail("inference is not requested"))
    prediction_to_replay_spec(CorePredictionConfig.model_validate(_public(state_cache=sizing)))


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"backend": "sglang"}, "backend"),
        ({"tp_size": True}, "tp_size"),
        ({"tp_size": 0}, "tp_size"),
        ({"pp_size": 2}, "PP=1"),
        ({"block_size": 1}, "block_size"),
        ({"kv_bytes_per_token": "16"}, "kv_bytes_per_token"),
        ({"kv_bytes_per_token": 1 << 63}, "overflows"),
        ({"num_speculative_tokens": -1}, "num_speculative_tokens"),
        ({"num_speculative_tokens": True}, "num_speculative_tokens"),
        ({"model_dtype": "fp8"}, "model_dtype"),
        ({"mamba_cache_dtype": "bfloat16"}, "mamba_cache_dtype"),
        ({"mamba_ssm_cache_dtype": "bfloat16"}, "mamba_ssm_cache_dtype"),
        ({"layout": []}, "layout"),
        ({"allow_hf_config_download": "false"}, "allow_hf_config_download"),
    ],
)
def test_public_state_estimator_validates_before_loading(overrides, error, monkeypatch):
    from aisimulate_core.sdk import estimate_state_cache, state_memory

    monkeypatch.setattr(state_memory, "_load_state_config", lambda *a, **k: pytest.fail("invalid inputs must not load"))
    args = {
        "backend": "vllm",
        "block_size": 64,
        "kv_bytes_per_token": 16,
        "allow_hf_config_download": True,
        **overrides,
    }
    with pytest.raises(ValueError, match=error):
        estimate_state_cache("must-not-load", **args)


@pytest.mark.parametrize(
    "sources,error",
    [
        ({}, "exactly one"),
        ({"model_path": "model", "model_config": {}}, "exactly one"),
        ({"model_config": []}, "mapping"),
        ({"model_path": ""}, "nonempty"),
    ],
)
def test_public_state_estimator_requires_one_model_source(sources, error):
    from aisimulate_core.sdk import estimate_state_cache

    with pytest.raises(ValueError, match=error):
        estimate_state_cache(**sources, backend="vllm", block_size=64, kv_bytes_per_token=16)


def test_public_state_estimator_download_is_opt_in(kda_model, tmp_path, monkeypatch):
    from aisimulate_core.sdk import estimate_state_cache, utils

    calls = []

    def download(name):
        calls.append(name)
        return kda_model[1]

    monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "_download_hf_config", download)
    with pytest.raises(ValueError, match="cannot load"):
        estimate_state_cache("uncached/model", backend="vllm", block_size=64, kv_bytes_per_token=16)
    assert calls == []
    assert (
        estimate_state_cache(
            "uncached/model", backend="vllm", block_size=64, kv_bytes_per_token=16, allow_hf_config_download=True
        )["bytes_per_request"]
        == 3072
    )
    assert calls == ["uncached/model"]


def test_public_state_estimator_checks_total_u64_overflow(kda_model):
    from aisimulate_core.sdk import estimate_state_cache

    with pytest.raises(ValueError, match="bytes_per_request"):
        estimate_state_cache(
            model_config=kda_model[1], backend="vllm", block_size=64, kv_bytes_per_token=((1 << 64) - 1) // 64
        )


@pytest.mark.parametrize("allow_download", [False, True])
def test_state_estimator_uses_cached_model_outside_fpm_default_set(tmp_path, kda_model, monkeypatch, allow_download):
    from aisimulate_core.sdk import estimate_state_cache, utils
    from aisimulate_core.sdk.common import DefaultHFModels

    model = "coverage/new-state-model"
    assert model not in DefaultHFModels
    (tmp_path / "coverage--new-state-model_config.json").write_text(json.dumps(kda_model[1]))
    monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "_download_hf_config", lambda *a: pytest.fail("cached configuration must not download"))
    result = estimate_state_cache(
        model, backend="vllm", block_size=64, kv_bytes_per_token=16, allow_hf_config_download=allow_download
    )
    assert result["bytes_per_request"] == 3072


def test_state_estimator_does_not_replace_invalid_cached_config(tmp_path, monkeypatch):
    from aisimulate_core.sdk import estimate_state_cache, utils

    (tmp_path / "coverage--invalid_config.json").write_text("{invalid json")
    monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "_download_hf_config", lambda *a: pytest.fail("invalid cached data must not fall back"))
    with pytest.raises(ValueError):
        estimate_state_cache(
            "coverage/invalid", backend="vllm", block_size=64, kv_bytes_per_token=16, allow_hf_config_download=True
        )


def test_qwen_gdn_is_not_in_kda_estimation_scope():
    from aisimulate_core.sdk import estimate_state_cache

    with pytest.raises(ValueError, match="unknown model geometry"):
        estimate_state_cache("Qwen/Qwen3.5-35B-A3B", backend="vllm", block_size=8192, kv_bytes_per_token=65536)
