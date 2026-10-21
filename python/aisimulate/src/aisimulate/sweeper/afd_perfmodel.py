# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance-model boundary for AFD A/F layer measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .afd_parallel import AFDInfeasible, AFDPhase, AFDReasonCategory, AFDTopology
from .replay import BackendDeploymentSpec

AFD_MEASUREMENT_API_VERSION = 1


@dataclass(frozen=True)
class AFDLayerTimes:
    """Performance-model measurements for one AFD phase and transformer layer."""

    phase: AFDPhase | str
    attention_ms: float
    ffn_ms: float
    a_to_f_ms: float
    f_to_a_ms: float
    num_layers: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            phase = self.phase if isinstance(self.phase, AFDPhase) else AFDPhase(self.phase)
        except ValueError as exc:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"phase must be prefill or decode, got {self.phase!r}",
            ) from exc
        if phase is AFDPhase.BOTH:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "AFDLayerTimes must describe one phase, not 'both'",
            )
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int) or self.num_layers < 1:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"num_layers must be a positive integer, got {self.num_layers!r}",
            )
        for name in ("attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_MEASUREMENT,
                    f"{name} must be a finite non-negative number, got {value!r}",
                    provenance={"field": name, "value": value},
                )
        if max(self.attention_ms, self.ffn_ms, self.a_to_f_ms, self.f_to_a_ms) <= 0:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "at least one AFD layer time must be positive",
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class AFDMeasurementRequest:
    """One concrete candidate and workload point to measure."""

    model_name: str
    hardware_sku: str
    backend: str
    backend_version: str
    topology: AFDTopology
    input_length: int
    output_length: int
    prefix: int = 0
    nextn: int = 0
    max_seq_len: int | None = None


@runtime_checkable
class AFDPerformanceModel(Protocol):
    """Provider of precise, per-layer A/F measurements."""

    def measure(self, request: AFDMeasurementRequest) -> Sequence[AFDLayerTimes]: ...


def _candidate_topology(sample: Mapping[str, Any]) -> AFDTopology:
    raw = dict(sample["afd"])
    # ``ffn_tp`` is a useful serialized derived value, not an AFDTopology
    # constructor field.
    raw.pop("ffn_tp", None)
    return AFDTopology(**raw)


def measurement_request_from_candidate(
    sample: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> AFDMeasurementRequest:
    """Build a measurement request, failing closed without concrete lengths."""

    input_length = workload.get("isl")
    output_length = workload.get("osl")
    if (
        isinstance(input_length, bool)
        or not isinstance(input_length, int)
        or input_length < 1
        or isinstance(output_length, bool)
        or not isinstance(output_length, int)
        or output_length < 1
    ):
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "AFD performance measurement requires concrete positive workload isl and osl; "
            "trace-only AFD measurement is not supported by this layer",
            provenance={"isl": input_length, "osl": output_length},
        )
    backend_version = sample.get("backend_version")
    if not isinstance(backend_version, str) or not backend_version:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "AFD performance measurement requires a resolved backend_version",
        )
    max_seq_len = sample.get("context_length")
    return AFDMeasurementRequest(
        model_name=str(sample["model_name"]),
        hardware_sku=str(sample["hardware_sku"]),
        backend=str(sample["backend"]),
        backend_version=backend_version,
        topology=_candidate_topology(sample),
        input_length=input_length,
        output_length=output_length,
        nextn=int(sample.get("aic_nextn") or 0),
        max_seq_len=int(max_seq_len) if max_seq_len is not None else None,
    )


class AICAFDPerformanceModel:
    """Measure A/F layer components through AIC's SDK."""

    def measure(self, request: AFDMeasurementRequest) -> tuple[AFDLayerTimes, ...]:
        # Imported lazily: the SDK pulls in the perf database and every model
        # definition, and this module is imported by the sweeper package root.
        from .afd_measure import measure_afd_layer_times

        try:
            return measure_afd_layer_times(
                model_path=request.model_name,
                system_name=request.hardware_sku,
                backend_name=request.backend,
                backend_version=request.backend_version,
                isl=request.input_length,
                osl=request.output_length,
                topology=request.topology,
                prefix=request.prefix,
                nextn=request.nextn,
                max_seq_len=request.max_seq_len,
            )
        except AFDInfeasible:
            # Already carries a specific category and provenance.
            raise
        except Exception as exc:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"AIC could not measure the AFD candidate: {type(exc).__name__}: {exc}",
                provenance={
                    "provider": "aic",
                    "model": request.model_name,
                    "hardware": request.hardware_sku,
                    "backend": request.backend,
                    "backend_version": request.backend_version,
                },
            ) from exc


def _measurement_payload(measurement: AFDLayerTimes) -> dict[str, Any]:
    return {
        "phase": measurement.phase.value,
        "attention_ms": float(measurement.attention_ms),
        "ffn_ms": float(measurement.ffn_ms),
        "a_to_f_ms": float(measurement.a_to_f_ms),
        "f_to_a_ms": float(measurement.f_to_a_ms),
        "num_layers": measurement.num_layers,
        "provenance": dict(measurement.provenance),
    }


def attach_afd_measurements(
    deployment: BackendDeploymentSpec,
    *,
    sample: Mapping[str, Any],
    workload: Mapping[str, Any],
    performance_model: AFDPerformanceModel,
) -> BackendDeploymentSpec:
    """Attach validated A/F measurements to an AFD deployment contract."""

    if deployment.deployment_mode not in {"afd", "afd+pd"}:
        return deployment
    request = measurement_request_from_candidate(sample, workload)
    measurements = tuple(performance_model.measure(request))
    expected = (
        {AFDPhase.PREFILL, AFDPhase.DECODE} if request.topology.phase is AFDPhase.BOTH else {request.topology.phase}
    )
    by_phase: dict[AFDPhase, AFDLayerTimes] = {}
    for measurement in measurements:
        if not isinstance(measurement, AFDLayerTimes):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "AFD performance model must return only AFDLayerTimes values",
            )
        if measurement.phase in by_phase:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"AFD performance model returned duplicate phase {measurement.phase.value!r}",
            )
        by_phase[measurement.phase] = measurement
    if set(by_phase) != expected:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "AFD performance model phase coverage does not match the candidate topology",
            provenance={
                "expected_phases": sorted(phase.value for phase in expected),
                "actual_phases": sorted(phase.value for phase in by_phase),
            },
        )

    metadata = deepcopy(deployment.performance_model_metadata)
    afd_metadata = dict(metadata.get("afd", {}))
    providers = {str(item.provenance.get("provider", "custom")) for item in measurements}
    afd_metadata.update(
        {
            "provider": providers.pop() if len(providers) == 1 else "mixed",
            "measurement_required": False,
            "measurement_api_version": AFD_MEASUREMENT_API_VERSION,
            "measurements": [
                _measurement_payload(by_phase[phase]) for phase in sorted(by_phase, key=lambda p: p.value)
            ],
        }
    )
    metadata["afd"] = afd_metadata
    return replace(deployment, performance_model_metadata=metadata)


__all__ = [
    "AFD_MEASUREMENT_API_VERSION",
    "AFDLayerTimes",
    "AFDMeasurementRequest",
    "AFDPerformanceModel",
    "AICAFDPerformanceModel",
    "attach_afd_measurements",
    "measurement_request_from_candidate",
]
