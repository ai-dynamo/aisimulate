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


def test_prefix_match_unit_requires_state_and_old_alignment_field_is_rejected():
    with pytest.raises(ValidationError, match="requires state_cache"):
        CorePredictionConfig.model_validate(_public(state_cache=None, prefix_match_unit=16))
    with pytest.raises(ValidationError, match="Extra inputs"):
        CorePredictionConfig.model_validate(_public(state_cache={**STATE, "prefill_block_size": 1536}))
