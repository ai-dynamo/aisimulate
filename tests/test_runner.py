# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-only implementation of the canonical Sweeper Runner contract."""

import json
import math
import pickle
from dataclasses import replace

import pytest
from pydantic import ValidationError

import aisimulate
from aisimulate import capacity as aic
from aisimulate.cli_args import build_parser
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config.traffic import SyntheticSource
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
from aisimulate.sweeper.config import Workload
from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH as DEEPSEEK_V41_MODEL_PATH

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


class PowerRecordingRuntime(RecordingRuntime):
    def run_replay_json(self, execution_spec_json):
        payload = json.loads(super().run_replay_json(execution_spec_json))
        payload.update({"power_w": 487.5, "power_coverage": 0.95})
        return json.dumps(payload)


class WithheldPowerRecordingRuntime(RecordingRuntime):
    def run_replay_json(self, execution_spec_json):
        payload = json.loads(super().run_replay_json(execution_spec_json))
        payload["power_coverage"] = 0.42
        return json.dumps(payload)


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
        "timing_model": timing or {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
    }


def _spec(*, deployment=None, workload=None, goal=None, concurrency=None, adapters=None):
    return ReplaySpec(
        backend_deployment=deployment
        or BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=_engine_args(),
            num_workers=2,
        ),
        workload=workload or {"isl": 8, "osl": 2, "concurrency": 1, "num_request_ratio": 1},
        goal=goal or {"target": "throughput"},
        concurrency=concurrency,
        adapters=adapters or {},
    )


def test_native_replay_provider_rejects_direct_only_graph_profile_before_compilation():
    from aisimulate import _runtime

    # The model and system deliberately do not exist: the replay provider must
    # reject this selector before constructing or extracting a native engine.
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "missing-graph-model",
            "system": "missing-graph-system",
            "backend": "sglang",
            "estimator_config": {"op_level": {"prefill_graph_profile": "sglang_glm52_nvfp4_vr200_tp4_graph_v1"}},
        },
    }
    args = {"worker_type": "aggregated", "block_size": 4, "num_gpu_blocks": 16, "timing_model": timing}
    # Nested native rank configuration reaches the provider without Python's
    # canonical capacity preflight, which independently rejects graph replay.
    args = {"rank": args}
    deployment = BackendDeploymentSpec(
        deployment_mode="agg", backend="sglang", backend_version="test", agg_engine_args=args, num_workers=1
    )
    with pytest.raises(RuntimeError, match="replay/scheduler timing is unqualified"):
        EngineReplayRunnerFactory(runtime=_runtime).create(0).run(_spec(deployment=deployment))


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
    assert capabilities.supports_trace_format("weka")
    assert capabilities.supports_trace_format("agentic_mooncake")
    assert capabilities.supports_agentic_lanes
    assert capabilities.supported_agentic_topologies == ("agg", "disagg")
    assert capabilities.supported_agentic_backends == ("vllm", "sglang")
    assert not capabilities.supports_agentic_host_offload
    assert not capabilities.supports_agentic_speculative_decoding
    assert capabilities.agentic_qualification == "functional_only"


@pytest.mark.parametrize("trace_format", ["weka", "agentic_mooncake", "dynamo"])
@pytest.mark.parametrize("nested_rank", [False, True])
@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ({"native_host_offload": {"num_host_blocks": 8}}, "HBM-only"),
        ({"aic_nextn": 1}, "speculative decoding disabled"),
        ({"nextn": 1}, "speculative decoding disabled"),
    ],
)
def test_agentic_capabilities_reject_unqualified_memory_and_decode_modes(
    trace_format, nested_rank, role, unsupported, message
):
    runtime = RecordingRuntime()
    args = _engine_args(role=role) | unsupported
    if nested_rank:
        args = {"rank": args}
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode="agg" if role == "aggregated" else "disagg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=args if role == "aggregated" else None,
            prefill_engine_args=args if role == "prefill" else _engine_args(role="prefill"),
            decode_engine_args=args if role == "decode" else _engine_args(role="decode"),
            num_workers=1,
        ),
        workload={"source_type": "trace", "trace_format": trace_format, "agentic_lanes": 1},
    )

    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec is None


@pytest.mark.parametrize("trace_format", ["weka", "agentic_mooncake", "dynamo"])
@pytest.mark.parametrize("topology", ["agg", "disagg"])
def test_agentic_capabilities_reject_trtllm(trace_format, topology):
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode=topology,
            backend="trtllm",
            backend_version="test",
            agg_engine_args=_engine_args(backend="trtllm"),
            num_workers=1,
        ),
        workload={"source_type": "trace", "trace_format": trace_format, "agentic_lanes": 1},
    )
    with pytest.raises(ValueError, match="agentic execution with backend 'trtllm'"):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)


def test_standard_dynamo_defers_agentic_restrictions_until_trace_kind_is_known():
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="test",
            agg_engine_args=_engine_args() | {"aic_nextn": 1},
            num_workers=1,
        ),
        workload={"source_type": "trace", "trace_format": "dynamo"},
    )
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)


def test_agentic_qualification_survives_default_python_report():
    qualification = {
        "agentic_qualification": "functional_only",
        "agentic_input_format": "weka",
        "agentic_lanes": 1,
        "agentic_model_projection": {
            "policy": "project_to_configured_target",
            "source_models": ["source-model"],
            "target_model": "test-model",
        },
        "weka_nested_timestamp_basis": "absolute",
    }

    class QualifiedRuntime(RecordingRuntime):
        def run_replay_json(self, execution_spec_json):
            report = json.loads(super().run_replay_json(execution_spec_json))
            return json.dumps(report | qualification)

    report = EngineReplayRunnerFactory(runtime=QualifiedRuntime()).create(0).run(_spec())

    assert {key: report.metadata[key] for key in qualification} == qualification
    assert report.metadata["power"]["publication_status"] == "unavailable"
    assert report.metrics["completed_requests"] == 1
    assert "native_report" not in report.metadata


def test_runner_preserves_weka_lane_input_without_defaulting_source_block_size():
    runtime = RecordingRuntime()
    runner = EngineReplayRunnerFactory(runtime=runtime).create(worker_id=0)
    runner.run(
        _spec(
            workload={
                "source_type": "trace",
                "load_type": "trace_timestamps",
                "trace_path": "weka-corpus",
                "trace_paths": ["weka-corpus"],
                "trace_format": "weka",
                "arrival_speedup_ratio": 1.0,
                "agentic_lanes": 2,
            }
        )
    )

    traffic = runtime.execution_spec["traffic"]
    assert traffic["trace_format"] == "weka"
    assert traffic["agentic_lanes"] == 2
    assert traffic["execution_model"] == "test-model"
    assert "trace_block_size" not in traffic


@pytest.mark.parametrize("model_key", ["model", "model_path"])
def test_agentic_runner_reads_canonical_model_identity(model_key):
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("aic_model_path")
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
        performance_model_metadata={"aggregated": {"provider": "aic", "config": {model_key: "test-model"}}},
    )
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(
            deployment=deployment,
            workload={
                "source_type": "trace",
                "load_type": "trace_timestamps",
                "trace_path": "agentic.jsonl",
                "trace_format": "agentic_mooncake",
            },
        )
    )
    assert runtime.execution_spec["traffic"]["execution_model"] == "test-model"


@pytest.mark.parametrize("nested_rank", [False, True])
@pytest.mark.parametrize("payload", [{}, {"config": None}, {"config": []}, {"config": "invalid"}])
def test_runner_rejects_nonmapping_aic_timing_before_materialization(nested_rank, payload, monkeypatch):
    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", lambda **kwargs: pytest.fail("invalid timing reached capacity"))
    runtime = RecordingRuntime()
    args = _engine_args(timing={"type": "external", "provider": "aic", **payload})
    if nested_rank:
        args = {"rank": args}
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode="agg", backend="vllm", backend_version="test", agg_engine_args=args, num_workers=1
        )
    )
    with pytest.raises(ValueError, match="external AIC timing config must be a mapping"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec is None


def test_runner_rejects_agentic_execution_without_a_target_model():
    engine_args = _engine_args()
    engine_args.pop("aic_model_path")
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=1,
    )

    with pytest.raises(
        ValueError,
        match="agentic execution requires a configured target model",
    ):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                deployment=deployment,
                workload={
                    "source_type": "trace",
                    "load_type": "trace_timestamps",
                    "trace_path": "weka-corpus",
                    "trace_format": "weka",
                },
            )
        )


def test_prediction_compiler_carries_weka_agentic_lanes() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": ["weka-corpus"],
                    "format": "weka",
                    "nested_timestamp_basis": "relative",
                },
                "load": {
                    "type": "trace_timestamps",
                    "speedup": 2.0,
                    "agentic_lanes": 3,
                },
            },
            "engine": {
                "model": "example/model",
                "hardware": "h200_sxm",
                "context_length": 1024,
                "workers": {"aggregated": {}},
            },
        }
    )
    workload = prediction_to_replay_spec(config).workload

    assert workload["trace_format"] == "weka"
    assert workload["trace_block_size"] is None
    assert workload["arrival_speedup_ratio"] == 2.0
    assert workload["agentic_lanes"] == 3
    assert workload["weka_nested_timestamp_basis"] == "relative"


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_engine_capability_rejects_online_disaggregated_weka_before_runtime(backend: str) -> None:
    runtime = RecordingRuntime()
    spec = ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend=backend,
            backend_version="test",
            num_prefill_workers=1,
            num_decode_workers=1,
        ),
        execution_mode="online",
        workload={"source_type": "trace", "trace_format": "weka"},
        goal={},
    )

    with pytest.raises(ValueError, match="does not support execution mode 'online'"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec is None


@pytest.mark.parametrize("trace_format", ["weka", "agentic_mooncake", "dynamo"])
@pytest.mark.parametrize("model_source", ["engine_args", "metadata"])
@pytest.mark.parametrize("decode_model", [None, "different-model", "test-model"])
def test_disaggregated_agentic_requires_same_target_model(trace_format, model_source, decode_model):
    runtime = RecordingRuntime()
    prefill = _engine_args(role="prefill")
    decode = _engine_args(role="decode")
    metadata = {}
    if model_source == "metadata":
        prefill.pop("aic_model_path")
        decode.pop("aic_model_path")
        metadata = {
            role: {"provider": "aic", "config": {"model_path": model}}
            for role, model in [("prefill", "test-model"), ("decode", decode_model)]
        }
    else:
        decode["aic_model_path"] = decode_model
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend="vllm",
            backend_version="test",
            prefill_engine_args=prefill,
            decode_engine_args=decode,
            performance_model_metadata=metadata,
            num_prefill_workers=1,
            num_decode_workers=1,
        ),
        workload={"source_type": "trace", "trace_format": trace_format, "agentic_lanes": 1},
    )
    runner = EngineReplayRunnerFactory(runtime=runtime).create(0)
    if decode_model != "test-model":
        with pytest.raises(ValueError, match="prefill and decode must use the same configured target model"):
            runner.run(spec)
        assert runtime.execution_spec is None
    else:
        runner.run(spec)
        assert runtime.execution_spec["traffic"]["execution_model"] == "test-model"


@pytest.mark.parametrize("trace_format", ["weka", "agentic_mooncake", "dynamo", "standard_dynamo"])
@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("timing_model", ["test-model", "other-model"])
@pytest.mark.parametrize("declared_target", [False, True])
def test_disaggregated_agentic_checks_actual_aic_timing_model(trace_format, role, timing_model, declared_target):
    runtime = RecordingRuntime()
    roles = {name: _engine_args(role=name) for name in ("prefill", "decode")}
    roles[role]["timing_model"] = {"type": "external", "provider": "aic", "config": {"model": timing_model}}
    workload = {"source_type": "trace", "trace_format": trace_format, "agentic_lanes": 1}
    if trace_format == "standard_dynamo":
        workload = {"source_type": "trace", "trace_format": "dynamo"}
    spec = _spec(
        deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend="vllm",
            backend_version="test",
            prefill_engine_args=roles["prefill"],
            decode_engine_args=roles["decode"],
            performance_model_metadata=(
                {name: {"provider": "aic", "config": {"model": "test-model"}} for name in roles}
                if declared_target
                else {}
            ),
            num_prefill_workers=1,
            num_decode_workers=1,
        ),
        workload=workload,
    )
    runner = EngineReplayRunnerFactory(runtime=runtime).create(0)
    if timing_model != "test-model" and trace_format != "standard_dynamo":
        message = (
            f"agentic {role} AIC timing model must match"
            if declared_target
            else "agentic prefill and decode must use the same configured target model"
        )
        with pytest.raises(ValueError, match=message):
            runner.run(spec)
        assert runtime.execution_spec is None
    else:
        runner.run(spec)
        if declared_target or timing_model == "test-model":
            assert runtime.execution_spec["traffic"]["execution_model"] == "test-model"
        else:
            assert "execution_model" not in runtime.execution_spec["traffic"]
        assert runtime.execution_spec["spec"]["engine"][role]["rank"]["timing_model"]["config"]["model"] == timing_model


@pytest.mark.parametrize(
    ("workload", "message"),
    [
        (
            {"source_type": "trace", "trace_format": "mooncake", "agentic_lanes": 1},
            "requires weka",
        ),
        (
            {"source_type": "trace", "trace_format": "weka", "agentic_lanes": 0},
            "positive integer",
        ),
        (
            {"source_type": "trace", "trace_format": "weka", "agentic_lanes": True},
            "positive integer",
        ),
        (
            {
                "source_type": "trace",
                "trace_format": "mooncake",
                "weka_nested_timestamp_basis": "absolute",
            },
            "requires Weka input",
        ),
        (
            {
                "source_type": "trace",
                "trace_format": "weka",
                "weka_nested_timestamp_basis": "guess",
            },
            "must be 'auto', 'absolute', or 'relative'",
        ),
    ],
)
def test_engine_capability_rejects_invalid_agentic_lane_controls(workload: dict, message: str) -> None:
    capabilities = EngineReplayRunnerFactory().capabilities()

    with pytest.raises(ValueError, match=message):
        capabilities.require_compatible(_spec(workload=workload))


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
    assert report.metadata["power"]["publication_status"] == "unavailable"
    assert report.metrics["power_w"] is None
    assert report.metrics["power_coverage"] is None


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_engine_runner_preserves_independent_sla_bounds(field: str, bound: float) -> None:
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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(prediction_to_replay_spec(public))

    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    assert rank["kv_cache_bytes_per_token"] == 131_072
    assert rank["native_host_offload"] == {
        "num_host_blocks": 4096,
        "d2h_bandwidth_gbps": 7.0,
        "h2d_bandwidth_gbps": 38.0,
    }


@pytest.mark.parametrize("rank_only_controls", [False, True])
def test_public_cuda_graph_reservation_reaches_native_capacity(tmp_path, monkeypatch, rank_only_controls):
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
    import aisimulate_core
    from aisimulate_core.sdk import RustForwardPassPerfModel

    class Estimator:
        def __init__(self, config):
            self.config = json.loads(aisimulate_core.RustForwardPassPerfModel.normalize_config(json.dumps(config)))
            self.config.update(backend_version="0.24.0", estimation_mode="op_level", fallback_policy="deny")

        def diagnostics(self):
            return {"readiness": "ready", "provenance": {"config": self.config}}

        def close(self):
            pass

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", Estimator)
    public = CorePredictionConfig.from_yaml(path)
    spec = prediction_to_replay_spec(public)
    if rank_only_controls:
        timing = spec.backend_deployment.agg_engine_args["timing_model"]["config"]
        timing.pop("cuda_graph_reserved_bytes")
        timing.pop("gpu_memory_utilization")
    runtime = RecordingRuntime()

    assert public.engine.workers.aggregated is not None
    capacity = public.engine.workers.aggregated.kv_cache.capacity
    assert capacity.cuda_graph_reserved_bytes == reserved_bytes
    assert spec.backend_deployment.agg_engine_args["cuda_graph_reserved_bytes"] == reserved_bytes

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)

    engine = runtime.execution_spec["spec"]["engine"]
    assert engine["rank"]["num_gpu_blocks"] == 321
    assert engine["rank"]["timing_model"]["config"]["cuda_graph_reserved_bytes"] == reserved_bytes
    assert calls[0]["cuda_graph_reserved_bytes"] == reserved_bytes


@pytest.mark.parametrize(
    "backend,field,value", [("vllm", "prefill_schedule_interval", 4), ("sglang", "prefill_decode_interval", 20)]
)
def test_public_prefill_interval_reaches_native_execution_rank(tmp_path, backend, field, value):
    path = tmp_path / "prediction.yaml"
    path.write_text(
        f"""\
engine:
  mode: aggregated
  model: example/model
  hardware: h200_sxm
  backend: {backend}
  context_length: 4096
  workers:
    aggregated:
      parallelism:
        attention_data: 2
      scheduler:
        {field}: {value}
      kv_cache:
        block_size: 16
        capacity: {{type: fixed, blocks: 128}}
      timing: {{type: fixed, prefill_ms: 1.0, decode_ms: 1.0}}
""",
        encoding="utf-8",
    )
    runtime = RecordingRuntime()
    public = CorePredictionConfig.from_yaml(path)

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(prediction_to_replay_spec(public))

    assert public.engine.workers.aggregated is not None
    assert getattr(public.engine.workers.aggregated.scheduler, field) == value
    assert runtime.execution_spec["spec"]["engine"]["rank"][field] == value


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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


@pytest.mark.parametrize("dtype", [None, "gemm_dtype", "moe_dtype", "fmha_dtype", "kv_cache_dtype", "comm_dtype"])
def test_runner_keeps_capacity_estimation_independent_from_fixed_timing(monkeypatch, dtype):
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("num_gpu_blocks")
    engine_args["gpu_memory_utilization"] = 0.8
    if dtype:
        engine_args["aic_" + dtype] = "fp8"
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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

    rank = runtime.execution_spec["engine"]["rank"]
    assert rank["num_gpu_blocks"] == 321
    assert rank["timing_model"]["type"] == "fixed"
    assert "gpu_memory_utilization" not in rank
    assert calls[0]["gpu_memory_utilization"] == 0.8
    if dtype:
        assert calls[0][dtype] == "fp8"
        assert "aic_" + dtype not in rank


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


def test_runner_preserves_native_power_provenance_without_raw_report():
    report = EngineReplayRunnerFactory(runtime=PowerRecordingRuntime()).create(worker_id=7).run(_spec())

    assert "native_report" not in report.metadata
    assert report.metrics["power_w"] == 487.5
    assert report.metrics["power_coverage"] == 0.95
    assert report.metadata["power"] == {
        "source": "modeled",
        "scope": "active_forward_pass_per_gpu",
        "power_w_unit": "W",
        "coverage_gate": 0.9,
        "publication_status": "available",
    }


def test_runner_preserves_withheld_power_without_raw_report():
    report = EngineReplayRunnerFactory(runtime=WithheldPowerRecordingRuntime()).create(worker_id=7).run(_spec())

    assert "native_report" not in report.metadata
    assert report.metrics["power_coverage"] == 0.42
    assert report.metrics["power_w"] is None
    assert report.metadata["power"]["publication_status"] == "withheld"


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
    assert {request["arrival_time_ms"] for request in runtime.execution_spec["requests"]} == {0.0}


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
    assert [request["arrival_time_ms"] for request in runtime.execution_spec["requests"]] == [0.0, 2.5, 5.0]


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

    arrivals = [request["arrival_time_ms"] for request in execution_specs[0]["requests"]]
    assert arrivals == [request["arrival_time_ms"] for request in execution_specs[1]["requests"]]
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
        return [(request["input_tokens"], request["output_tokens"]) for request in execution_spec["requests"]]

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

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

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

    timing_config = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]
    assert timing_config["backend_version"] == "0.11.1"


def test_direct_replay_normalizes_both_version_aliases_without_mutating_input():
    runtime = RecordingRuntime()
    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "Qwen/Qwen3-32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "current",
            "tp": 2,
            "attention_dp": 1,
        },
    }
    args = {"tensor_parallel_size": 2, "rank": {"block_size": 4, "num_gpu_blocks": 128, "timing_model": timing}}
    deployment = BackendDeploymentSpec(
        deployment_mode="agg", backend="vllm", backend_version="current", agg_engine_args=args, num_workers=1
    )
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))
    actual = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]["backend_version"]
    from aisimulate_core.sdk.perf_database import resolve_query_version

    assert actual == resolve_query_version("h200_sxm", "vllm", "current")
    assert timing["config"]["backend_version"] == "current"


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(workload=workload))


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


def test_runner_threads_forward_model_alias_into_aic_timing():
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args["aic_forward_model"] = "fpm"
    engine_args["aic_fpm_parquet_path"] = "/artifacts/reviewed-fpm.parquet"
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.25.1",
        agg_engine_args=engine_args,
        num_workers=2,
    )

    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))

    rank = runtime.execution_spec["engine"]["rank"]
    assert rank["timing_model"]["config"]["forward_model"] == "fpm"
    assert rank["timing_model"]["config"]["fpm_parquet_path"] == "/artifacts/reviewed-fpm.parquet"
    assert "aic_forward_model" not in rank
    assert "aic_fpm_parquet_path" not in rank


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


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
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("worker_policy", [None, "SILICON", "SOL"])
def test_public_replay_keeps_decoder_profile_and_database_policy(replay, worker_policy, monkeypatch):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    class ReadyEstimator:
        def __init__(self, config):
            self.config = config

        def diagnostics(self):
            return {"readiness": "ready", "provenance": {"config": self.config}}

        def close(self):
            pass

    # This exercises configuration transport with a recording runtime, not
    # readiness or prediction for the model.
    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", ReadyEstimator)
    public = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": DEEPSEEK_V41_MODEL_PATH,
                "hardware": "gb300",
                "backend": "sglang",
                "decoder_replay": replay,
                "database_mode": "SILICON",
                "enable_shared_layer": False,
                "strict_provenance": True,
                "workers": {
                    "aggregated": {
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                        "timing": {"database_mode": worker_policy},
                    }
                },
            }
        }
    )
    runtime = RecordingRuntime()
    spec = prediction_to_replay_spec(public)
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    config = runtime.execution_spec["spec"]["engine"]["rank"]["timing_model"]["config"]
    assert config.get("decoder_replay", False) is replay
    assert config["database_mode"] == (worker_policy or "SILICON")
    assert config["enable_shared_layer"] is False
    assert config["strict_provenance"] is True
    metadata = spec.backend_deployment.performance_model_metadata["aggregated"]["config"]
    assert metadata.get("decoder_replay", False) is replay
    assert metadata["database_mode"] == config["database_mode"]


@pytest.mark.parametrize(
    ("model", "backend"),
    [("example/model", "sglang"), (DEEPSEEK_V41_MODEL_PATH, "vllm")],
)
def test_public_replay_rejects_unsupported_model_or_backend(model, backend):
    with pytest.raises(ValueError, match="decoder_replay requires"):
        CorePredictionConfig.model_validate(
            {
                "engine": {
                    "model": model,
                    "hardware": "gb300",
                    "backend": backend,
                    "decoder_replay": True,
                }
            }
        )


@pytest.mark.parametrize("field", ["aic_decoder_replay", "aic_enable_shared_layer", "aic_strict_provenance"])
def test_replay_policy_alias_requires_a_boolean(field):
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args[field] = "false"
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        parallel_config={"tp": 2, "attention_dp": 1, "replicas": 2},
        agg_engine_args=engine_args,
        num_workers=2,
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))


def test_memory_detail_reuses_capacity_calculation_without_changing_execution(monkeypatch):
    from aisimulate_core.sdk import memory

    calls = []
    estimate = {
        "source": "native",
        "total_gpu_capacity_bytes": 4096,
        "total_kv_size_bytes": 1024,
        "total_kv_size_tokens": 128,
        "kv_size_per_token_bytes": 8,
        "tolerance_adjusted": None,
        "memory_breakdown": {"weights_bytes": 2048, "activations_bytes": 512},
    }

    def estimate_kv(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(estimate)

    monkeypatch.setattr(memory, "estimate_kv_cache", estimate_kv)
    args = _engine_args()
    args.pop("num_gpu_blocks")
    args["aic_backend_version"] = "test"
    deployment = BackendDeploymentSpec(
        deployment_mode="agg", backend="vllm", backend_version="test", agg_engine_args=args, num_workers=1
    )
    plain_runtime, detail_runtime = RecordingRuntime(), RecordingRuntime()
    plain = EngineReplayRunnerFactory(runtime=plain_runtime).create(0).run(_spec(deployment=deployment))
    detailed = (
        EngineReplayRunnerFactory(runtime=detail_runtime)
        .create(0)
        .run(
            _spec(deployment=deployment),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_memory_diagnostics=True),
        )
    )
    assert len(calls) == 2  # One estimate per replay, with identical estimator arguments.
    assert calls[0] == calls[1]
    assert plain_runtime.execution_spec == detail_runtime.execution_spec
    assert plain.metrics == detailed.metrics
    data = detailed.metadata["native_report"]["memory_diagnostics"]["aggregated"]
    assert data["status"] == "available"
    assert data["scope"] == "capacity_estimate_per_rank"
    assert data["memory_breakdown"] == estimate["memory_breakdown"]
    assert data["estimated_num_gpu_blocks"] == detail_runtime.execution_spec["engine"]["rank"]["num_gpu_blocks"] == 32


def test_memory_detail_with_explicit_blocks_does_not_guess_components():
    runtime = RecordingRuntime()
    report = (
        EngineReplayRunnerFactory(runtime=runtime)
        .create(0)
        .run(
            _spec(),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_memory_diagnostics=True),
        )
    )
    data = report.metadata["native_report"]["memory_diagnostics"]["aggregated"]
    assert data["status"] == "unavailable"
    assert "memory_breakdown" not in data


def test_native_report_memory_error_becomes_host_resource_failure(monkeypatch):
    import sys
    from types import ModuleType

    class HostResourceError(RuntimeError):
        pass

    resources = ModuleType("aisimulate.resources")
    resources.ResourceLimitError = HostResourceError
    monkeypatch.setitem(sys.modules, "aisimulate.resources", resources)

    class LimitedRuntime:
        def run_replay_json(self, payload):
            raise MemoryError("report storage: No space left on device")

    with pytest.raises(HostResourceError, match="report storage: No space left on device"):
        EngineReplayRunnerFactory(runtime=LimitedRuntime()).create(0).run(_spec())


@pytest.mark.parametrize(
    "watts,coverage",
    [
        (500.0, 0.89),
        (500.0, None),
        (-1.0, 1.0),
        (0.0, 1.0),
        (None, 1.1),
        (None, -0.1),
        (True, 1.0),
        (None, True),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (-math.inf, 1.0),
        (None, math.nan),
        (None, math.inf),
        (None, -math.inf),
        (10**400, 1.0),
        (None, 10**400),
    ],
)
def test_runner_rejects_invalid_power_publication(watts, coverage):
    from aisimulate.runner import _normalize_engine_replay_report

    with pytest.raises(InvalidRunnerError):
        _normalize_engine_replay_report({"power_w": watts, "power_coverage": coverage}, include_native_report=False)


def test_runner_rejects_overflowing_ordinary_metric():
    from aisimulate.runner import _normalize_engine_replay_report

    with pytest.raises(InvalidRunnerError, match="output_throughput_tok_s.*not finite"):
        _normalize_engine_replay_report({"output_throughput_tok_s": 10**400}, include_native_report=False)


@pytest.mark.parametrize("trace_format", ["weka", "agentic_mooncake", "dynamo"])
@pytest.mark.parametrize("warmup", [False, True])
def test_agentic_snapshot_reaches_native_payload_and_default_python_evidence(trace_format: str, warmup: bool) -> None:
    evidence = [{"seed": 2**64 - 1, "lane_id": "lane:0", "t_star_ms": 50.0}]
    phases = {"schema": "aisimulate.agentic.phases.v1", "phase": "profile"}

    class SnapshotRuntime(RecordingRuntime):
        def run_replay_json(self, execution_spec_json):
            report = json.loads(super().run_replay_json(execution_spec_json))
            return json.dumps(report | {"agentic_snapshots": evidence} | ({"agentic_phases": phases} if warmup else {}))

    runtime = SnapshotRuntime()
    report = (
        EngineReplayRunnerFactory(runtime=runtime)
        .create(0)
        .run(
            _spec(
                workload={
                    "source_type": "trace",
                    "load_type": "trace_timestamps",
                    "trace_path": "corpus",
                    "trace_format": trace_format,
                    "agentic_lanes": 1,
                    "agentic_snapshot": {"seed": 2**64 - 1},
                    "agentic_warmup": warmup,
                }
            )
        )
    )
    assert runtime.execution_spec["traffic"]["agentic_snapshot"] == {"seed": 2**64 - 1}
    assert report.metadata["agentic_snapshots"] == evidence
    assert runtime.execution_spec["traffic"]["agentic_warmup"] is warmup
    assert report.metadata.get("agentic_phases") == (phases if warmup else None)
    assert "native_report" not in report.metadata


@pytest.mark.parametrize("snapshot", [{}, {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"seed": 1, "extra": 0}])
def test_agentic_snapshot_runner_rejects_untyped_invalid_options(snapshot: dict) -> None:
    runtime = RecordingRuntime()
    with pytest.raises(ValueError, match="agentic_snapshot"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            _spec(
                workload={
                    "source_type": "trace",
                    "load_type": "trace_timestamps",
                    "trace_path": "corpus",
                    "trace_format": "weka",
                    "agentic_lanes": 1,
                    "agentic_snapshot": snapshot,
                }
            )
        )
    assert runtime.execution_spec is None


@pytest.mark.parametrize(
    "capability,label", [("supports_agentic_snapshots", "snapshots"), ("supports_agentic_warmup", "warmup")]
)
def test_agentic_snapshot_capability_is_explicit(capability: str, label: str) -> None:
    from dataclasses import replace

    factory = EngineReplayRunnerFactory()
    assert getattr(factory.capabilities(), capability)
    capabilities = replace(factory.capabilities(), **{capability: False})
    with pytest.raises(ValueError, match=f"does not support agentic {label}"):
        capabilities.require_compatible(
            _spec(
                workload={
                    "source_type": "trace",
                    "load_type": "trace_timestamps",
                    "trace_path": "corpus",
                    "trace_format": "weka",
                    "agentic_lanes": 1,
                    "agentic_snapshot": {"seed": 42},
                    "agentic_warmup": True,
                }
            )
        )


@pytest.mark.parametrize("warmup", [None, 0, 1, "true", {}])
def test_agentic_warmup_runner_rejects_untyped_invalid_options(warmup) -> None:
    runtime = RecordingRuntime()
    with pytest.raises(ValueError, match="agentic_warmup must be a boolean"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(workload={"agentic_warmup": warmup}))
    assert runtime.execution_spec is None


def test_agentic_warmup_runner_requires_snapshot() -> None:
    runtime = RecordingRuntime()
    with pytest.raises(ValueError, match="agentic_warmup requires agentic_snapshot"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(workload={"agentic_warmup": True}))
    assert runtime.execution_spec is None


@pytest.mark.parametrize("sampler", ["numpy_random_state", "python_random"])
def test_runner_honors_length_sampler(sampler):
    # Fixed vectors from the benchmark's NumPy RandomState contract and the
    # legacy Python sampler. The entire input vector is drawn first.
    expected = {
        "numpy_random_state": [(8, 4), (9, 4), (8, 5), (9, 4), (9, 4)],
        "python_random": [(9, 5), (9, 5), (8, 5), (9, 5), (10, 5)],
    }
    for _ in range(2):
        runtime = RecordingRuntime()
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            _spec(
                workload={
                    "isl": 10,
                    "osl": 5,
                    "request_count": 5,
                    "arrival_interval_ms": 0.0,
                    "random_range_ratio": 0.8,
                    "random_seed": 0,
                    "length_sampler": sampler,
                }
            )
        )
        assert [(r["input_tokens"], r["output_tokens"]) for r in runtime.execution_spec["requests"]] == expected[
            sampler
        ]


def test_runner_rejects_unknown_length_sampler():
    with pytest.raises(ValueError, match="length_sampler"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "isl": 10,
                    "osl": 5,
                    "request_count": 2,
                    "arrival_interval_ms": 0.0,
                    "length_sampler": "typo",
                }
            )
        )


@pytest.mark.parametrize(
    "sampler,seed",
    [("numpy_random_state", 0xFFFF_FFFF), ("python_random", 0x1_0000_0000), ("python_random", 0xFFFF_FFFF_FFFF_FFFF)],
)
def test_runner_sampler_seed_upper_bounds(sampler, seed):
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(
            workload={
                "isl": 10,
                "osl": 5,
                "request_count": 2,
                "arrival_interval_ms": 0.0,
                "random_range_ratio": 0.8,
                "random_seed": seed,
                "length_sampler": sampler,
            }
        )
    )
    assert len(runtime.execution_spec["requests"]) == 2


def test_runner_rejects_numpy_seed_above_uint32():
    with pytest.raises(ValueError, match="numpy_random_state random_seed.*32-bit"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                workload={
                    "isl": 10,
                    "osl": 5,
                    "request_count": 2,
                    "arrival_interval_ms": 0.0,
                    "random_seed": 0x1_0000_0000,
                    "length_sampler": "numpy_random_state",
                }
            )
        )


@pytest.mark.parametrize("sampler", ["numpy_random_state", "typo"])
def test_runner_rejects_nondefault_sampler_for_trace(sampler):
    with pytest.raises(ValueError, match="length_sampler only applies to synthetic replay"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(workload={"trace_path": "unused.jsonl", "length_sampler": sampler})
        )


@pytest.mark.parametrize("schema", [Workload, SyntheticSource])
def test_length_sampler_is_not_exposed_by_public_workload_schemas(schema):
    with pytest.raises(ValidationError) as error:
        schema.model_validate({"length_sampler": "numpy_random_state"})
    assert any(e["loc"] == ("length_sampler",) and e["type"] == "extra_forbidden" for e in error.value.errors())


def test_length_sampler_is_not_exposed_by_public_cli(capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["predict", "-c", "unused.yaml", "--length-sampler", "numpy_random_state"])
    assert error.value.code == 2
    assert "unrecognized arguments: --length-sampler numpy_random_state" in capsys.readouterr().err


@pytest.mark.parametrize("source_type", ["synthetic", "trace"])
@pytest.mark.parametrize("sampler", ["numpy_random_state", "python_random", "typo"])
@pytest.mark.parametrize("deployment_mode", ["agg", "disagg", "afd", "afd+pd"])
def test_workload_driver_rejects_length_sampler_before_native_execution(source_type, sampler, deployment_mode):
    runtime = RecordingRuntime()
    spec = _spec(workload={"source_type": source_type, "length_sampler": sampler})
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, deployment_mode=deployment_mode))
    with pytest.raises(
        ValueError, match="length_sampler requires materialized direct synthetic replay without source_type"
    ):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    assert runtime.execution_spec is None


@pytest.mark.parametrize("layout", ["flat", "null_flat", "canonical", "canonical_fallback", "nested"])
@pytest.mark.parametrize("version", ["", "next", "literal_next"])
def test_direct_replay_capacity_uses_the_resolved_timing_version(monkeypatch, tmp_path, layout, version):
    from aisimulate import _runtime
    from aisimulate_core.sdk import perf_database

    slots = perf_database.get_version_slots("h200_sxm", "vllm")
    expected = slots["next" if version == "literal_next" else version or "current"]
    if version == "literal_next":
        version = expected
    assert slots["next"] != slots["current"]
    phase = "python"
    consumed_versions = {"python": [], "native": []}
    get_database = perf_database.get_database

    def recorded_database(*args, **kwargs):
        database = get_database(*args, **kwargs)
        consumed_versions[phase].append(database.version)
        return database

    monkeypatch.setattr(perf_database, "get_database", recorded_database)

    class NativeRuntime:
        def run_replay_json(self, raw):
            nonlocal phase
            phase = "native"
            # Native replay builds timing first, copies its resolved provenance
            # into AicTimingConfig, then passes that version to capacity.
            return _runtime.run_replay_json(raw)

    timing = {
        "type": "external",
        "provider": "aic",
        "config": {
            "model": "Qwen/Qwen3-32B-FP8",
            "system": "h200_sxm",
            "backend": "vllm",
            "tp": 2,
            "attention_dp": 1,
        },
    }
    args = {
        "aic_backend": "vllm",
        "aic_model_path": "Qwen/Qwen3-32B-FP8",
        "aic_system": "h200_sxm",
        "aic_tp_size": 2,
        "block_size": 64,
    }
    if layout == "null_flat":
        args["aic_backend_version"] = None
    elif layout in {"canonical", "canonical_fallback"}:
        timing["config"]["estimation_mode"] = "op_level"
        timing["config"]["worker_type"] = "aggregated"
        if layout == "canonical_fallback":
            # The first root has no database. Timing selects the bundled root;
            # capacity must retain that resolved choice for every version pin.
            timing["config"]["systems_paths"] = [str(tmp_path), "default"]
        args = {"tensor_parallel_size": 2, "block_size": 64, "timing_model": timing}
    elif layout == "nested":
        args = {"tensor_parallel_size": 2, "rank": {"block_size": 64, "timing_model": timing}}
    original = json.loads(json.dumps(args))
    deployment = BackendDeploymentSpec(
        deployment_mode="agg", backend="vllm", backend_version=version, agg_engine_args=args, num_workers=1
    )
    report = EngineReplayRunnerFactory(runtime=NativeRuntime()).create(0).run(_spec(deployment=deployment))

    assert report.metrics["completed_requests"] == 1
    assert set(consumed_versions["native"]) == {expected}
    assert set(consumed_versions["python"]) == (set() if layout == "nested" else {expected})
    assert args == original


@pytest.mark.parametrize("root_location", ["role", "timing"])
def test_direct_replay_resolves_aliases_in_the_effective_systems_root(monkeypatch, tmp_path, root_location):
    from pathlib import Path

    import yaml

    from aisimulate_core.sdk import perf_database

    bundled = Path(perf_database.__file__).resolve().parents[1] / "systems"
    system = yaml.safe_load((bundled / "h200_sxm.yaml").read_text())
    system["data_dir"] = str(bundled / system["data_dir"])
    (tmp_path / "h200_sxm.yaml").write_text(yaml.safe_dump(system))
    custom_version = "0.25.0"
    (tmp_path / "query_versions.yaml").write_text(yaml.safe_dump({"defaults": {"vllm": {"current": custom_version}}}))
    assert perf_database.get_version_slots("h200_sxm", "vllm")["current"] != custom_version
    assert perf_database.get_version_slots("h200_sxm", "vllm", systems_paths=str(tmp_path))["current"] == custom_version
    get_database = perf_database.get_database
    versions = []

    def recorded_database(*args, **kwargs):
        database = get_database(*args, **kwargs)
        versions.append(database.version)
        return database

    monkeypatch.setattr(perf_database, "get_database", recorded_database)
    args = {
        "aic_backend": "vllm",
        "aic_model_path": "Qwen/Qwen3-32B-FP8",
        "aic_system": "h200_sxm",
        "aic_tp_size": 2,
        "block_size": 64,
    }
    if root_location == "role":
        args["systems_path"] = str(tmp_path)
    else:
        args["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": {
                "model": "Qwen/Qwen3-32B-FP8",
                "system": "h200_sxm",
                "backend": "vllm",
                "tp": 2,
                "attention_dp": 1,
                "systems_paths": [str(tmp_path)],
                "systems_path": "default",
            },
        }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg", backend="vllm", backend_version="current", agg_engine_args=args, num_workers=1
    )
    report = EngineReplayRunnerFactory().create(0).run(_spec(deployment=deployment))

    assert report.metrics["completed_requests"] == 1
    assert set(versions) == {custom_version}
    assert "aic_backend_version" not in args


@pytest.mark.parametrize("inferred_capacity", [False, True])
def test_flat_engine_controls_reach_native_timing_identity(inferred_capacity, monkeypatch):
    controls = {"enable_eplb": True, "wideep_num_slots": 128, "moe_backend": "deepep_moe", "attention_backend": "fa3"}
    seen = {}

    def estimate(**kwargs):
        seen.update(kwargs)
        return 16

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    args = _engine_args()
    args.pop("timing_model")
    if inferred_capacity:
        args.pop("num_gpu_blocks")
    args.update({"aic_" + name: value for name, value in controls.items()})
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(
        _spec(
            deployment=BackendDeploymentSpec(
                deployment_mode="agg",
                backend="vllm",
                backend_version="test",
                agg_engine_args=args,
                num_workers=1,
            )
        )
    )
    rank = runtime.execution_spec["engine"]["rank"]
    assert {name: rank["timing_model"]["config"][name] for name in controls} == controls
    assert not {"aic_" + name for name in controls} & rank.keys()
    if inferred_capacity:
        assert {name: seen[name] for name in controls} == controls


@pytest.mark.parametrize(
    "name,value", [("enable_eplb", 1), ("wideep_num_slots", 0), ("wideep_num_slots", True), ("moe_backend", False)]
)
def test_flat_engine_controls_validate_types(name, value):
    args = _engine_args()
    args.pop("timing_model")
    args["aic_" + name] = value
    runtime = RecordingRuntime()
    with pytest.raises(ValueError, match=name):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            _spec(
                deployment=BackendDeploymentSpec(
                    deployment_mode="agg",
                    backend="vllm",
                    backend_version="test",
                    agg_engine_args=args,
                    num_workers=1,
                )
            )
        )
    assert runtime.execution_spec is None


def test_flat_controls_cannot_override_inactive_nested_identity():
    args = _engine_args(timing={"type": "external", "provider": "aic", "config": {"enable_eplb": False}})
    args["aic_enable_eplb"] = True
    with pytest.raises(ValueError, match="configured both"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                deployment=BackendDeploymentSpec(
                    deployment_mode="agg",
                    backend="vllm",
                    backend_version="test",
                    agg_engine_args=args,
                    num_workers=1,
                )
            )
        )


def test_flat_active_controls_reject_fixed_timing():
    args = {**_engine_args(), "aic_enable_eplb": True}
    with pytest.raises(ValueError, match="require an AIC timing"):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(
            _spec(
                deployment=BackendDeploymentSpec(
                    deployment_mode="agg",
                    backend="vllm",
                    backend_version="test",
                    agg_engine_args=args,
                    num_workers=1,
                )
            )
        )


@pytest.mark.parametrize("alias", ["decode_workload_distribution", "aic_decode_workload_distribution"])
@pytest.mark.parametrize("selected", [None, "observed_glm52_nvfp4_decode_1ab2c747975e_v1"])
def test_runner_threads_optional_decode_profile_alias(alias, selected):
    runtime = RecordingRuntime()
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args[alias] = selected
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=2,
    )
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(_spec(deployment=deployment))
    timing = runtime.execution_spec["engine"]["rank"]["timing_model"]["config"]
    if selected is None:
        assert "decode_workload_distribution" not in timing
    else:
        assert timing["decode_workload_distribution"] == selected


@pytest.mark.parametrize(
    "extras",
    [
        {
            "decode_workload_distribution": "",
        },
        {
            "decode_workload_distribution": 7,
        },
        {"decode_workload_distribution": "one", "aic_decode_workload_distribution": "two"},
    ],
)
def test_runner_rejects_invalid_or_duplicate_decode_profile_alias(extras):
    engine_args = _engine_args()
    engine_args.pop("timing_model")
    engine_args.update(extras)
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=engine_args,
        num_workers=2,
    )
    with pytest.raises(ValueError):
        EngineReplayRunnerFactory(runtime=RecordingRuntime()).create(0).run(_spec(deployment=deployment))
