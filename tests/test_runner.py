# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-only implementation of the canonical Sweeper Runner contract."""

import json
import pickle

import pytest

import aisimulate
from aisimulate import aic
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.replay.config import ReplayCliConfig, ReplayOutputConfig
from aisimulate.runner import (
    EngineReplayRunner,
    EngineReplayRunnerFactory,
    InvalidRunnerError,
)
from aisimulate.sweeper import (
    AdapterReplaySpec,
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RuntimeHookSpec,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class RecordingRuntime:
    def __init__(self):
        self.execution_spec = None
        self.execution_spec_json = None

    def run_replay_json(self, execution_spec_json):
        self.execution_spec_json = execution_spec_json
        self.execution_spec = json.loads(execution_spec_json)
        return json.dumps(
            {
                "duration_ms": 4.0,
                "output_throughput_tok_s": 2000.0,
                "gpu_hours": 0.001,
                "mean_ttft_ms": 2.0,
                "mean_tpot_ms": 1.0,
                "mean_e2e_latency_ms": 4.0,
                "mean_output_token_throughput_per_user": 1000.0,
                "goodput_output_throughput_tok_s": 1500.0,
                "completed_requests": 1,
            }
        )


def _engine_args(*, role="aggregated", backend="vllm", timing=None):
    return {
        "worker_type": role,
        "engine_type": backend,
        "aic_backend": backend,
        "aic_model_path": "test-model",
        "aic_system": "test-system",
        "aic_tp_size": 2,
        "aic_attention_dp_size": 1,
        "block_size": 4,
        "num_gpu_blocks": 16,
        "timing_model": timing
        or {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
    }


def _spec(
    *, deployment=None, workload=None, goal=None, concurrency=None, adapters=None
):
    return ReplaySpec(
        backend_deployment=deployment
        or BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=_engine_args(),
            num_workers=2,
        ),
        workload=workload
        or {"isl": 8, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
        goal=goal or {"target": "throughput"},
        concurrency=concurrency,
        adapters=adapters or {},
    )


def test_public_namespace_exports_engine_runner_contract():
    assert aisimulate.EngineReplayRunner is EngineReplayRunner
    assert aisimulate.EngineReplayRunnerFactory is EngineReplayRunnerFactory


def test_legacy_replay_config_preserves_execution_mode() -> None:
    config = ReplayCliConfig(
        trace_files=(),
        extra_engine_args={"engine_type": "vllm"},
        prefill_engine_args=None,
        decode_engine_args=None,
        num_workers=1,
        num_prefill_workers=0,
        num_decode_workers=0,
        replay_mode="online",
        workload={"isl": 8, "osl": 2, "request_count": 1, "concurrency": 1},
        goal={},
        output=ReplayOutputConfig(),
    )

    assert config.to_replay_spec().execution_mode == "online"


def test_factory_is_pickleable_and_advertises_engine_only_capabilities():
    factory = pickle.loads(pickle.dumps(EngineReplayRunnerFactory()))
    capabilities = factory.capabilities()

    assert capabilities.supports_backend_topology("vllm", "agg")
    assert capabilities.supports_backend_topology("sglang", "disagg")
    assert capabilities.supports_backend_topology("trtllm", "disagg")
    assert capabilities.supports_disaggregated_attention_dp
    assert capabilities.supported_execution_modes == ("offline",)
    assert capabilities.supported_hooks == ()


def test_runner_lowers_canonical_spec_and_returns_replay_report():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=7)

    report = runner.run(_spec())

    assert isinstance(report, ReplayReport)
    assert report.metrics["output_throughput_tok_s"] == 2000.0
    assert report.metrics["mean_ttft_ms"] == 2.0
    assert report.metrics["mean_tpot_ms"] == 1.0
    assert report.metrics["mean_e2e_latency_ms"] == 4.0
    assert report.metrics["mean_output_token_throughput_per_user"] == 1000.0
    assert report.metrics["goodput_output_throughput_tok_s"] == 1500.0
    execution = runtime.execution_spec
    assert execution["topology"] == {
        "kind": "aggregated",
        "workers": {"initial_workers": 2, "startup_delay_ms": 0.0},
    }
    assert execution["engine"]["tensor_parallel_size"] == 2
    assert execution["engine"]["num_gpu_blocks_is_explicit"] is True
    assert execution["engine"]["rank"]["backend"] == "vllm"
    assert execution["requests"][0]["input_tokens"] == 8
    assert execution["record_per_request"] is False
    assert isinstance(runtime.execution_spec_json, str)
    assert report.metadata == {}


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_engine_runner_preserves_independent_sla_bounds(
    field: str, bound: float
) -> None:
    runtime = RecordingRuntime()
    spec = _spec(
        goal={
            "target": "throughput",
            "strict_sla": False,
            "sla": {field: bound},
        }
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    assert runtime.execution_spec["sla"] == {field: bound}


def test_runner_lowers_sglang_with_prefix_caching_disabled():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.update(
        {
            "engine_type": "sglang",
            "aic_backend": "sglang",
            "block_size": 1,
            "enable_prefix_caching": False,
        }
    )
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="sglang",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    assert runtime.execution_spec["engine"]["rank"]["enable_prefix_caching"] is False


def test_runner_preserves_native_host_offload_rank_config():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args["kv_transfer_bytes_per_token"] = 333
    engine_args["kv_cache_bytes_per_token"] = 131_072
    engine_args["native_host_offload"] = {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 7.0,
        "h2d_bandwidth_gbps": 38.0,
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    rank = runtime.execution_spec["engine"]["rank"]
    assert rank["kv_transfer_bytes_per_token"] == 333
    assert rank["kv_cache_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == engine_args["native_host_offload"]


def test_public_host_offload_config_reaches_native_execution_rank():
    runtime = RecordingRuntime()
    public = CorePredictionConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "replicas": 1,
                            "tensor": 1,
                            "pipeline": 1,
                            "attention_data": 1,
                            "moe_tensor": 1,
                            "moe_expert": 1,
                        },
                        "scheduler": {
                            "max_batched_tokens": 8192,
                            "max_sequences": 4,
                        },
                        "kv_cache": {
                            "block_size": 16,
                            "prefix_caching": True,
                            "bytes_per_token": 131_072,
                            "capacity": {"type": "fixed", "blocks": 128},
                            "host_offload": {
                                "num_host_blocks": 4096,
                                "d2h_bandwidth_gbps": 7.0,
                                "h2d_bandwidth_gbps": 38.0,
                            },
                        },
                        "timing": {
                            "type": "fixed",
                            "prefill_ms": 1.0,
                            "decode_ms": 1.0,
                        },
                    }
                },
            }
        }
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        prediction_to_replay_spec(public)
    )

    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    assert rank["kv_cache_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 7.0,
        "h2d_bandwidth_gbps": 38.0,
    }


def test_public_cuda_graph_reservation_reaches_native_capacity(tmp_path, monkeypatch):
    reserved_bytes = 14_559_947_612
    path = tmp_path / "prediction.yaml"
    path.write_text(
        f"""\
engine:
  mode: aggregated
  model: example/model
  hardware: h200_sxm
  backend: vllm
  context_length: 4096
  workers:
    aggregated:
      kv_cache:
        block_size: 16
        capacity:
          type: default
          memory_fraction: 0.8
          cuda_graph_reserved_bytes: {reserved_bytes}
""",
        encoding="utf-8",
    )
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 321

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    public = CorePredictionConfig.from_yaml(path)
    spec = prediction_to_replay_spec(public)
    runtime = RecordingRuntime()

    assert public.engine.workers.aggregated is not None
    capacity = public.engine.workers.aggregated.kv_cache.capacity
    assert capacity.cuda_graph_reserved_bytes == reserved_bytes
    assert (
        spec.backend_deployment.agg_engine_args["cuda_graph_reserved_bytes"]
        == reserved_bytes
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    engine = runtime.execution_spec["spec"]["engine"]
    assert engine["rank"]["num_gpu_blocks"] == 321
    assert (
        engine["rank"]["timing_model"]["config"]["cuda_graph_reserved_bytes"]
        == reserved_bytes
    )
    assert calls[0]["cuda_graph_reserved_bytes"] == reserved_bytes


def test_public_prefill_schedule_interval_reaches_native_execution_rank(tmp_path):
    path = tmp_path / "prediction.yaml"
    path.write_text(
        """\
engine:
  mode: aggregated
  model: example/model
  hardware: h200_sxm
  backend: vllm
  context_length: 4096
  workers:
    aggregated:
      parallelism:
        attention_data: 2
      scheduler:
        prefill_schedule_interval: 4
      kv_cache:
        block_size: 16
        capacity: {type: fixed, blocks: 128}
      timing: {type: fixed, prefill_ms: 1.0, decode_ms: 1.0}
""",
        encoding="utf-8",
    )
    runtime = RecordingRuntime()
    public = CorePredictionConfig.from_yaml(path)

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        prediction_to_replay_spec(public)
    )

    assert public.engine.workers.aggregated is not None
    assert public.engine.workers.aggregated.scheduler.prefill_schedule_interval == 4
    assert (
        runtime.execution_spec["spec"]["engine"]["rank"]["prefill_schedule_interval"]
        == 4
    )


def test_runner_materializes_aic_capacity_before_native_execution(monkeypatch):
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("num_gpu_blocks")
    engine_args.pop("timing_model")
    engine_args["aic_backend_version"] = "test"
    engine_args["aic_nextn"] = 3
    engine_args["aic_pp_size"] = 2
    engine_args["gpu_memory_utilization"] = 0.8
    engine_args["cuda_graph_reserved_bytes"] = 14559947612
    engine_args["systems_path"] = "/tmp/custom-systems.yaml"
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 321

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    assert runtime.execution_spec["engine"]["rank"]["num_gpu_blocks"] == 321
    assert runtime.execution_spec["engine"]["num_gpu_blocks_is_explicit"] is False
    timing_config = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]
    assert timing_config["pp"] == 2
    assert timing_config["systems_path"] == "/tmp/custom-systems.yaml"
    assert timing_config["gpu_memory_utilization"] == 0.8
    assert timing_config["cuda_graph_reserved_bytes"] == 14559947612
    assert calls[0]["pp_size"] == 2
    assert calls[0]["systems_path"] == "/tmp/custom-systems.yaml"
    assert calls[0]["gpu_memory_utilization"] == 0.8
    assert calls[0]["cuda_graph_reserved_bytes"] == 14559947612
    assert "cuda_graph_reserved_bytes" not in runtime.execution_spec["engine"]["rank"]
    assert "nextn" not in calls[0]


def test_runner_rejects_nested_inferred_capacity_when_fixed_timing_discards_reservation():
    engine_args = {
        "engine_type": "vllm",
        "aic_backend": "vllm",
        "aic_model_path": "test-model",
        "aic_system": "test-system",
        "rank": {
            "backend": "vllm",
            "block_size": 4,
            "cuda_graph_reserved_bytes": 1 << 30,
            "timing_model": {
                "type": "fixed",
                "prefill_ms": 2.0,
                "decode_ms": 1.0,
            },
        },
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    with pytest.raises(ValueError, match="requires an AIC timing model"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


def test_runner_keeps_capacity_estimation_independent_from_fixed_timing(monkeypatch):
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("num_gpu_blocks")
    engine_args["gpu_memory_utilization"] = 0.8
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 321

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        parallel_config={"tp": 2, "attention_dp": 1, "replicas": 1},
        agg_engine_args=engine_args,
        num_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    rank = runtime.execution_spec["engine"]["rank"]
    assert rank["num_gpu_blocks"] == 321
    assert rank["timing_model"]["type"] == "fixed"
    assert "gpu_memory_utilization" not in rank
    assert calls[0]["gpu_memory_utilization"] == 0.8


def test_runner_captures_requested_raw_and_per_request_report():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=7)

    report = runner.run(
        _spec(),
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )

    assert runtime.execution_spec["record_per_request"] is True
    assert report.metadata["native_report"]["completed_requests"] == 1


def test_engine_runner_rejects_unsupported_telemetry_before_runtime_invocation():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=7)

    with pytest.raises(
        InvalidRunnerError,
        match="JSON runtime does not yet expose replay telemetry",
    ):
        runner.run(
            _spec(),
            output_requirements=ReplayOutputRequirements(capture_telemetry=True),
        )

    assert runtime.execution_spec_json is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "must be a JSON string"),
        ("not-json", "returned invalid report JSON"),
        ("[]", "must be a JSON object"),
    ],
)
def test_runner_rejects_invalid_runtime_json_boundary_results(payload, message):
    class InvalidRuntime:
        def run_replay_json(self, execution_spec_json):
            assert isinstance(execution_spec_json, str)
            return payload

    with pytest.raises(InvalidRunnerError, match=message):
        EngineReplayRunnerFactory(runtime=InvalidRuntime()).create(0).run(_spec())


def test_runner_preserves_closed_loop_concurrency_in_execution_spec():
    runtime = RecordingRuntime()
    spec = _spec(
        workload={"isl": 4, "osl": 1, "concurrency": 3, "num_request_ratio": 2},
        concurrency=3,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    assert runtime.execution_spec["max_in_flight"] == 3
    assert len(runtime.execution_spec["requests"]) == 6
    assert {
        request["arrival_time_ms"] for request in runtime.execution_spec["requests"]
    } == {0.0}


def test_runner_materializes_fixed_interval_open_loop_requests():
    runtime = RecordingRuntime()
    spec = _spec(
        workload={
            "isl": 4,
            "osl": 1,
            "request_count": 3,
            "arrival_interval_ms": 2.5,
        }
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    assert runtime.execution_spec["max_in_flight"] is None
    assert [
        request["arrival_time_ms"] for request in runtime.execution_spec["requests"]
    ] == [0.0, 2.5, 5.0]


def test_runner_materializes_seeded_poisson_open_loop_requests():
    execution_specs = []
    for _ in range(2):
        runtime = RecordingRuntime()
        spec = _spec(
            workload={
                "isl": 4,
                "osl": 1,
                "request_count": 4,
                "request_rate": 2.0,
                "arrival_seed": 17,
            }
        )
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
        execution_specs.append(runtime.execution_spec)

    arrivals = [
        request["arrival_time_ms"] for request in execution_specs[0]["requests"]
    ]
    assert arrivals == [
        request["arrival_time_ms"] for request in execution_specs[1]["requests"]
    ]
    assert arrivals[0] == 0.0
    assert arrivals == sorted(arrivals)


def test_runner_randomizes_synthetic_lengths_deterministically():
    execution_specs = []
    for seed in (7, 7, 8):
        runtime = RecordingRuntime()
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            _spec(
                workload={
                    "isl": 100,
                    "osl": 50,
                    "request_count": 32,
                    "arrival_interval_ms": 0.0,
                    "random_range_ratio": 0.8,
                    "random_seed": seed,
                }
            )
        )
        execution_specs.append(runtime.execution_spec)

    def lengths(execution_spec):
        return [
            (request["input_tokens"], request["output_tokens"])
            for request in execution_spec["requests"]
        ]

    first_lengths = lengths(execution_specs[0])
    assert first_lengths == lengths(execution_specs[1])
    assert first_lengths != lengths(execution_specs[2])
    assert len(set(first_lengths)) > 1
    assert all(80 <= isl <= 100 and 40 <= osl <= 50 for isl, osl in first_lengths)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_runner_rejects_invalid_random_range_ratio(ratio):
    with pytest.raises(ValueError, match="random_range_ratio"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "isl": 64,
                    "osl": 2,
                    "request_count": 2,
                    "arrival_interval_ms": 0.0,
                    "random_range_ratio": ratio,
                }
            )
        )


def test_runner_rejects_random_length_options_for_trace_replay():
    with pytest.raises(ValueError, match="only apply to synthetic replay"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "trace_path": "unused.jsonl",
                    "random_range_ratio": 0.8,
                }
            )
        )


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_runner_lowers_disaggregated_grouped_engines(backend):
    runtime = RecordingRuntime()
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend=backend,
        backend_version="test",
        prefill_engine_args=_engine_args(role="prefill", backend=backend),
        decode_engine_args=_engine_args(role="decode", backend=backend),
        num_prefill_workers=2,
        num_decode_workers=3,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    assert runtime.execution_spec["topology"]["kind"] == "disaggregated"
    assert runtime.execution_spec["topology"]["prefill"]["initial_workers"] == 2
    assert runtime.execution_spec["topology"]["decode"]["initial_workers"] == 3
    assert set(runtime.execution_spec["engine"]) == {"prefill", "decode"}
    assert runtime.execution_spec["engine"]["prefill"]["rank"]["backend"] == backend
    assert runtime.execution_spec["engine"]["decode"]["rank"]["backend"] == backend


@pytest.mark.parametrize(
    ("prefill_dp", "decode_dp"),
    [(2, 1), (1, 2), (2, 4), (2, 2)],
)
def test_runner_lowers_disaggregated_attention_dp(prefill_dp, decode_dp):
    runtime = RecordingRuntime()
    prefill_args = _engine_args(role="prefill")
    decode_args = _engine_args(role="decode")
    prefill_args["aic_attention_dp_size"] = prefill_dp
    decode_args["aic_attention_dp_size"] = decode_dp
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="vllm",
        backend_version="test",
        parallel_config={
            "prefill_tp": 2,
            "prefill_attention_dp": prefill_dp,
            "prefill_replicas": 1,
            "decode_tp": 2,
            "decode_attention_dp": decode_dp,
            "decode_replicas": 1,
        },
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=1,
        num_decode_workers=1,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    engine = runtime.execution_spec["engine"]
    assert engine["prefill"]["dp_size"] == prefill_dp
    assert engine["decode"]["dp_size"] == decode_dp


def test_runner_threads_canonical_backend_version_into_aic_timing():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        parallel_config={"tp": 2, "attention_dp": 1, "replicas": 2},
        agg_engine_args=engine_args,
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    timing = runtime.execution_spec["engine"]["rank"]["timing_model"]
    assert timing["config"]["backend_version"] == "0.11.1"


def test_runner_accepts_matching_backend_version_in_explicit_aic_timing():
    runtime = RecordingRuntime()
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 2,
            "attention_dp": 1,
            "backend_version": "0.11.1",
        },
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        agg_engine_args=_engine_args(timing=timing),
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    timing_config = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]
    assert timing_config["backend_version"] == "0.11.1"


def test_runner_rejects_conflicting_backend_version_in_explicit_aic_timing():
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 2,
            "attention_dp": 1,
            "backend_version": "0.10.0",
        },
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.1",
        agg_engine_args=_engine_args(timing=timing),
        num_workers=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"timing_model\.config\.backend_version='0\.10\.0' conflicts with "
            r"BackendDeploymentSpec backend_version='0\.11\.1'"
        ),
    ):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


def test_runner_rejects_parallel_config_that_conflicts_with_engine_args():
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        parallel_config={"tp": 4, "replicas": 2},
        agg_engine_args=_engine_args(),
        num_workers=2,
    )

    with pytest.raises(ValueError, match="parallel_config.tp=4 conflicts"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("turns_per_session", 2),
        ("shared_prefix_ratio", 0.5),
        ("num_prefix_groups", 2),
        ("inter_turn_delay_ms", 10.0),
    ],
)
def test_engine_runner_fails_closed_for_unimplemented_synthetic_shapes(field, value):
    workload = {
        "isl": 8,
        "osl": 2,
        "concurrency": 1,
        "num_request_ratio": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(workload=workload)
        )


def test_engine_runner_does_not_silently_parse_a_dynamo_trace_as_mooncake():
    with pytest.raises(ValueError, match="supports only format='mooncake'"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "trace_path": "unused.jsonl",
                    "trace_format": "dynamo",
                }
            )
        )


def test_engine_runner_rejects_dynamo_runtime_hooks():
    hook = RuntimeHookSpec(
        provider="dynamo.router",
        kind="placement_policy",
        api_version=1,
        config={"router_mode": "kv_router", "router_config": {}},
    )
    spec = _spec(
        adapters={
            "dynamo.router": AdapterReplaySpec(runtime_hooks=(hook,)),
        }
    )

    with pytest.raises(ValueError, match="does not support runtime hook"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(spec)


def test_runner_rejects_nested_backend_that_conflicts_with_deployment():
    engine_args = _engine_args()
    engine_args["rank"] = {
        "backend": "sglang",
        "block_size": 1,
        "num_gpu_blocks": 16,
        "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
    }
    for field in (
        "block_size",
        "num_gpu_blocks",
        "timing_model",
    ):
        engine_args.pop(field)

    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    with pytest.raises(ValueError, match="rank backend conflicts"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


def test_runner_threads_forward_model_alias_into_aic_timing():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args["aic_forward_model"] = "fpm"
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.25.1",
        agg_engine_args=engine_args,
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(deployment=deployment)
    )

    rank = runtime.execution_spec["engine"]["rank"]
    assert rank["timing_model"]["config"]["forward_model"] == "fpm"
    assert "aic_forward_model" not in rank


def test_runner_rejects_forward_model_on_rank_and_in_explicit_aic_timing():
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "test-model",
            "backend": "vllm",
            "system": "test-system",
            "tp": 2,
            "attention_dp": 1,
            "forward_model": "fpm",
        },
    }
    engine_args = _engine_args(timing=timing)
    engine_args["aic_forward_model"] = "fpm"
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=2,
    )

    with pytest.raises(
        ValueError,
        match=r"configured both on the rank and inside timing_model\.config: forward_model",
    ):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )


@pytest.mark.parametrize("value", ["layerwise", "", 3])
def test_runner_rejects_unknown_forward_model(value):
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args["aic_forward_model"] = value
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=2,
    )

    with pytest.raises(ValueError, match="forward_model"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(deployment=deployment)
        )
