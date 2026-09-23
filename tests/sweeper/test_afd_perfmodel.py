# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD performance-model adapter and replay metadata tests."""

from types import SimpleNamespace

import pytest

from aisimulate.sweeper.afd_parallel import AFDInfeasible, AFDPhase, AFDTopology
from aisimulate.sweeper.afd_perfmodel import (
    AFDLayerTimes,
    AICAFDPerformanceModel,
    attach_afd_measurements,
    measurement_request_from_candidate,
)
from aisimulate.sweeper.replay import BackendDeploymentSpec


def _topology(*, phase: str = "decode") -> AFDTopology:
    return AFDTopology(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=4,
        tp_a=2,
        a_batch_size=16,
        phase=phase,
        combined_with_pd=False,
        comm_overhead_factor=1.5,
    )


def _sample(topology: AFDTopology) -> dict:
    return {
        "deployment_mode": "afd",
        "model_name": "example/model",
        "hardware_sku": "example_sku",
        "backend": "vllm",
        "backend_version": "test-version",
        "context_length": 4096,
        "afd": topology.provenance()["topology"],
    }


def test_aic_provider_extracts_precise_layer_measurement_without_double_calibration(monkeypatch):
    from aisimulate.sweeper import afd_measure

    captured = {}

    class FakeSession:
        def __init__(self, *, afd_config, **kwargs):
            captured["afd_config"] = afd_config

        def run_afd(self, runtime_config, **kwargs):
            captured["runtime_config"] = runtime_config
            captured["run_afd_kwargs"] = kwargs
            return SimpleNamespace(
                check_oom=lambda: False,
                get_result_dict=lambda: {
                    "afd_layer_measurements": {
                        "decode": {
                            "attention_ms": 1.23456,
                            "ffn_ms": 2.34567,
                            "a_to_f_ms": 0.12345,
                            "f_to_a_ms": 0.23456,
                            "num_layers": 32,
                        }
                    }
                },
            )

    monkeypatch.setattr(
        afd_measure,
        "_load_database",
        lambda system, backend, version: (
            SimpleNamespace(system_spec={"node": {"num_gpus_per_node": 4}}),
            "test-version",
        ),
    )
    monkeypatch.setattr(afd_measure, "get_backend", lambda name: SimpleNamespace(name=name))
    for resolver in ("resolve_context_fmha_by_data", "resolve_dsv4_moe_arch", "resolve_nvfp4_for_system"):
        monkeypatch.setattr(afd_measure, resolver, lambda *args, **kwargs: None)
    monkeypatch.setattr(afd_measure, "AFDInferenceSession", FakeSession)

    request = measurement_request_from_candidate(
        _sample(_topology()),
        {"isl": 128, "osl": 32},
    )
    (measurement,) = AICAFDPerformanceModel().measure(request)

    assert measurement.phase is AFDPhase.DECODE
    assert measurement.attention_ms == 1.23456
    assert measurement.a_to_f_ms == 0.12345
    assert measurement.num_layers == 32
    assert measurement.provenance["backend_version"] == "test-version"
    # The candidate's own comm_overhead_factor (1.5 above) must not be baked into
    # the layer measurement: the backend-neutral evaluator applies it exactly once.
    assert captured["afd_config"].comm_overhead_factor == 1.0
    assert captured["afd_config"].combined_with_pd is False
    assert captured["runtime_config"].batch_size == 32
    # The measurement layer deliberately exposes no KV-cache budget knob:
    # run_afd resolves the A-worker fraction from the backend default (vLLM by
    # version, SGLang 0.88, TRT-LLM 0.9), keeping the OOM verdict on the same
    # footing as the agg/disagg sweeps. Passing any value here would bypass
    # that resolution for every candidate.
    assert "free_gpu_memory_fraction" not in captured["run_afd_kwargs"]
    # Measurements are raw cost-side inputs: no speculative projection is
    # requested from run_afd at this layer. Acceptance progress belongs to
    # the evaluation/replay layer once AFD-MTP lands (see the afd_measure
    # docstring).
    assert "speculative_profile" not in captured["run_afd_kwargs"]


def test_aic_provider_measures_mtp_cost_side_without_projection(monkeypatch):
    from aisimulate.sweeper import afd_measure

    captured = {}

    class FakeSession:
        def __init__(self, *, a_model_config, f_model_config, **kwargs):
            captured["a_model_config"] = a_model_config
            captured["f_model_config"] = f_model_config

        def run_afd(self, runtime_config, **kwargs):
            captured["run_afd_kwargs"] = kwargs
            return SimpleNamespace(
                check_oom=lambda: False,
                get_result_dict=lambda: {
                    "afd_layer_measurements": {
                        "decode": {
                            "attention_ms": 1.0,
                            "ffn_ms": 2.0,
                            "a_to_f_ms": 0.1,
                            "f_to_a_ms": 0.2,
                            "num_layers": 32,
                        }
                    }
                },
            )

    monkeypatch.setattr(
        afd_measure,
        "_load_database",
        lambda system, backend, version: (
            SimpleNamespace(system_spec={"node": {"num_gpus_per_node": 4}}),
            "test-version",
        ),
    )
    monkeypatch.setattr(afd_measure, "get_backend", lambda name: SimpleNamespace(name=name))
    for resolver in ("resolve_context_fmha_by_data", "resolve_dsv4_moe_arch", "resolve_nvfp4_for_system"):
        monkeypatch.setattr(afd_measure, resolver, lambda *args, **kwargs: None)
    monkeypatch.setattr(afd_measure, "AFDInferenceSession", FakeSession)

    # nextn arrives through the sample's pinned aic_nextn, bypassing the config
    # gates (which live in SearchSpace/EnginePredictionConfig validation): this
    # is the direct-API surface the ongoing AFD-MTP work builds on.
    request = measurement_request_from_candidate(
        {**_sample(_topology()), "aic_nextn": 2},
        {"isl": 128, "osl": 32},
    )
    (measurement,) = AICAFDPerformanceModel().measure(request)

    # The draft depth is written into both pool configs (cost side), while no
    # acceptance projection is requested from run_afd: layer times stay raw.
    assert captured["a_model_config"].nextn == 2
    assert captured["f_model_config"].nextn == 2
    assert "speculative_profile" not in captured["run_afd_kwargs"]
    assert measurement.attention_ms == 1.0


def test_aic_provider_wraps_unexpected_measurement_errors(monkeypatch):
    from aisimulate.sweeper import afd_measure

    def boom(**kwargs):
        raise RuntimeError("database exploded")

    monkeypatch.setattr(afd_measure, "measure_afd_layer_times", boom)

    request = measurement_request_from_candidate(
        _sample(_topology()),
        {"isl": 128, "osl": 32},
    )
    with pytest.raises(AFDInfeasible, match="could not measure the AFD candidate"):
        AICAFDPerformanceModel().measure(request)


def test_attach_measurements_replaces_unresolved_metadata():
    topology = _topology()
    sample = _sample(topology)
    deployment = BackendDeploymentSpec(
        deployment_mode="afd",
        backend="vllm",
        backend_version="test-version",
        performance_model_metadata={"afd": {"provider": "unresolved", "measurement_required": True}},
    )

    class Provider:
        def measure(self, request):
            return (
                AFDLayerTimes(
                    phase="decode",
                    attention_ms=1.0,
                    ffn_ms=2.0,
                    a_to_f_ms=0.1,
                    f_to_a_ms=0.2,
                    num_layers=32,
                    provenance={"provider": "test"},
                ),
            )

    measured = attach_afd_measurements(
        deployment,
        sample=sample,
        workload={"isl": 128, "osl": 32},
        performance_model=Provider(),
    )

    assert deployment.performance_model_metadata["afd"]["provider"] == "unresolved"
    assert measured.performance_model_metadata["afd"]["provider"] == "test"
    assert measured.performance_model_metadata["afd"]["measurement_required"] is False
    assert measured.performance_model_metadata["afd"]["measurements"][0]["phase"] == "decode"


def test_measurement_boundary_rejects_trace_without_concrete_lengths():
    with pytest.raises(AFDInfeasible, match="concrete positive workload isl and osl"):
        measurement_request_from_candidate(
            _sample(_topology()),
            {"trace_path": "/tmp/trace.jsonl"},
        )


def test_attach_measurements_rejects_wrong_phase_coverage():
    topology = _topology(phase="both")
    sample = _sample(topology)
    deployment = BackendDeploymentSpec(
        deployment_mode="afd",
        backend="vllm",
        backend_version="test-version",
    )

    class Provider:
        def measure(self, request):
            return (
                AFDLayerTimes(
                    phase="decode",
                    attention_ms=1.0,
                    ffn_ms=2.0,
                    a_to_f_ms=0.1,
                    f_to_a_ms=0.2,
                    num_layers=32,
                ),
            )

    with pytest.raises(AFDInfeasible, match="phase coverage"):
        attach_afd_measurements(
            deployment,
            sample=sample,
            workload={"isl": 128, "osl": 32},
            performance_model=Provider(),
        )
