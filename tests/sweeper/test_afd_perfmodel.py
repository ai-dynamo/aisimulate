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


def test_aic_provider_extracts_precise_layer_measurement_without_double_calibration():
    calls = []

    def estimator(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            raw={
                "afd_layer_measurements": {
                    "decode": {
                        "attention_ms": 1.23456,
                        "ffn_ms": 2.34567,
                        "a_to_f_ms": 0.12345,
                        "f_to_a_ms": 0.23456,
                        "num_layers": 32,
                    }
                }
            }
        )

    request = measurement_request_from_candidate(
        _sample(_topology()),
        {"isl": 128, "osl": 32},
    )
    (measurement,) = AICAFDPerformanceModel(estimator).measure(request)

    assert measurement.phase is AFDPhase.DECODE
    assert measurement.attention_ms == 1.23456
    assert measurement.a_to_f_ms == 0.12345
    assert calls[0][1]["comm_overhead_factor"] == 1.0
    assert calls[0][1]["backend_version"] == "test-version"


def test_attach_measurements_replaces_unresolved_metadata():
    topology = _topology()
    sample = _sample(topology)
    deployment = BackendDeploymentSpec(
        deployment_mode="afd",
        backend="vllm",
        backend_version="test-version",
        performance_model_metadata={
            "afd": {"provider": "unresolved", "measurement_required": True}
        },
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
    assert (
        measured.performance_model_metadata["afd"]["measurements"][0]["phase"]
        == "decode"
    )


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
