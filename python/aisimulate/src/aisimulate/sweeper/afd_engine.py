# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground execution of one staged AFD full pass."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .afd_parallel import (
    AFD_SCHEMA_VERSION,
    AFDInfeasible,
    AFDPhase,
    AFDPipelineModel,
    AFDReasonCategory,
    AFDTopology,
)
from .afd_perfmodel import AFDLayerTimes


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            f"{name} must be a positive integer, got {value!r}",
        )


def _positive_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            f"{name} must be a positive finite number, got {value!r}",
        )


@dataclass(frozen=True)
class AFDPhaseEvaluation:
    """One phase's A/F pipeline evaluation for foreground execution."""

    phase: AFDPhase
    step_latency_ms: float
    sequence_rate: float
    tokens_per_second: float
    communication_hidden: bool
    balance_ratio: float
    cycle_ms: float
    pipeline_fill_ms: float
    requested_pipeline_model: AFDPipelineModel
    effective_pipeline_model: AFDPipelineModel
    total_gpus: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def evaluate_afd_phase(
    topology: AFDTopology,
    times: AFDLayerTimes,
    *,
    input_length: int,
    output_length: int,
    latency_correction: float = 1.0,
) -> AFDPhaseEvaluation:
    """Evaluate one AFD phase with the legacy K=3/K=2/serial formulas."""

    _positive_int("input_length", input_length)
    _positive_int("output_length", output_length)
    _positive_finite("latency_correction", latency_correction)
    if topology.phase is not AFDPhase.BOTH and topology.phase is not times.phase:
        raise AFDInfeasible(
            AFDReasonCategory.INCOMPATIBLE_PHASE,
            f"topology phase={topology.phase.value!r} cannot evaluate measurement phase={times.phase.value!r}",
            provenance=topology.provenance(),
        )

    t_a = float(times.attention_ms)
    t_f = float(times.ffn_ms)
    t_a2f = float(times.a_to_f_ms) * topology.comm_overhead_factor
    t_f2a = float(times.f_to_a_ms) * topology.comm_overhead_factor
    t_c = t_a2f + t_f2a
    requested = topology.pipeline_model
    effective = requested
    hidden = False
    if requested is AFDPipelineModel.SERIAL:
        cycle = t_a + t_a2f + t_f + t_f2a
    elif requested is AFDPipelineModel.CONSERVATIVE:
        cycle = max(t_a + t_a2f, t_f + t_f2a)
    else:
        min_microbatches = 2.0 + t_c / max(t_a, t_f, 1e-9)
        if topology.num_microbatches < min_microbatches:
            effective = AFDPipelineModel.CONSERVATIVE
            cycle = max(t_a + t_a2f, t_f + t_f2a)
        else:
            cycle = max(t_a, t_f, t_c)
            hidden = t_c <= max(t_a, t_f)
    fill = t_a + t_f + t_a2f + t_f2a
    step = (
        fill + cycle * max(topology.num_microbatches * times.num_layers - 1, 0)
    ) * float(latency_correction)
    if step <= 0:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "AFD pipeline evaluation produced a non-positive step latency",
        )
    if times.phase is AFDPhase.DECODE:
        tokens_per_second = topology.total_batch_size / (step / 1000.0)
        sequence_rate = tokens_per_second / output_length
    else:
        sequence_rate = topology.total_batch_size / (step / 1000.0)
        tokens_per_second = sequence_rate * input_length
    return AFDPhaseEvaluation(
        phase=times.phase,
        step_latency_ms=step,
        sequence_rate=sequence_rate,
        tokens_per_second=tokens_per_second,
        communication_hidden=hidden,
        balance_ratio=min(t_a, t_f) / max(t_a, t_f, 1e-9),
        cycle_ms=cycle,
        pipeline_fill_ms=fill,
        requested_pipeline_model=requested,
        effective_pipeline_model=effective,
        total_gpus=topology.total_gpus,
        provenance={
            "schema_version": AFD_SCHEMA_VERSION,
            "formula": "AFDInferenceSession._pipeline_global_step_latency",
            "layer_times": {
                "attention_ms": t_a,
                "ffn_ms": t_f,
                "a_to_f_ms": t_a2f,
                "f_to_a_ms": t_f2a,
                "num_layers": times.num_layers,
            },
            "measurement": dict(times.provenance),
            "topology": topology.provenance(),
            "latency_correction": latency_correction,
        },
    )


class AFDStage(str, Enum):
    """One resource stage in the A-to-F-to-A layer cycle."""

    ATTENTION = "attention"
    A_TO_F = "a_to_f"
    FFN = "ffn"
    F_TO_A = "f_to_a"


@dataclass(frozen=True)
class AFDStageInterval:
    """One stage's modeled occupancy in a full-pass schedule."""

    stage: AFDStage
    layer: int
    microbatch: int
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class AFDForegroundPass:
    """An eagerly planned, non-preemptive AFD full pass."""

    pass_id: int
    phase: AFDPhase
    started_at_ms: float
    end_ms: float
    input_length: int
    output_length: int
    evaluation: AFDPhaseEvaluation
    intervals: tuple[AFDStageInterval, ...]


@dataclass(frozen=True)
class AFDForegroundCompletion:
    """Effects released at an AFD full-pass completion boundary."""

    pass_id: int
    phase: AFDPhase
    completed_at_ms: float
    completed_sequences: int
    completed_tokens: int
    pass_latency_ms: float


def _validate_time(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite non-negative number, got {value!r}")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number, got {value!r}")
    return normalized


class AFDForegroundEngine:
    """Plan and complete one AFD A/F pass at a time.

    The engine makes every A -> transfer -> F -> transfer stage explicit for
    every ``(layer, microbatch)`` unit. Unit starts are separated by the exact
    cadence selected by :func:`evaluate_afd_phase`, so the final interval ends
    at the same full-pass boundary as the backend-neutral legacy formula.
    """

    def __init__(self, topology: AFDTopology, measurements: tuple[AFDLayerTimes, ...]) -> None:
        if not isinstance(topology, AFDTopology):
            raise TypeError("topology must be an AFDTopology")
        by_phase: dict[AFDPhase, AFDLayerTimes] = {}
        for measurement in measurements:
            if not isinstance(measurement, AFDLayerTimes):
                raise TypeError("measurements must contain only AFDLayerTimes")
            if measurement.phase in by_phase:
                raise AFDInfeasible(
                    AFDReasonCategory.INVALID_MEASUREMENT,
                    f"duplicate AFD foreground measurement for phase {measurement.phase.value!r}",
                )
            by_phase[measurement.phase] = measurement
        expected = {AFDPhase.PREFILL, AFDPhase.DECODE} if topology.phase is AFDPhase.BOTH else {topology.phase}
        if set(by_phase) != expected:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                "AFD foreground measurements do not match topology phase coverage",
                provenance={
                    "expected_phases": sorted(phase.value for phase in expected),
                    "actual_phases": sorted(phase.value for phase in by_phase),
                },
            )
        self._topology = topology
        self._measurements = by_phase
        self._next_pass_id = 0
        self._in_flight: AFDForegroundPass | None = None

    @property
    def in_flight(self) -> AFDForegroundPass | None:
        return self._in_flight

    def execute_pass(
        self,
        *,
        phase: AFDPhase | str,
        now_ms: float,
        input_length: int,
        output_length: int,
        latency_correction: float = 1.0,
    ) -> AFDForegroundPass:
        """Eagerly plan one non-preemptive pass and retain it until completion."""

        now = _validate_time(now_ms, "pass start time")
        if self._in_flight is not None:
            raise RuntimeError(f"AFD foreground pass {self._in_flight.pass_id} is already in flight")
        try:
            resolved_phase = AFDPhase(phase)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported AFD foreground phase {phase!r}") from exc
        if resolved_phase is AFDPhase.BOTH:
            raise ValueError("execute_pass requires one concrete prefill or decode phase")
        measurement = self._measurements.get(resolved_phase)
        if measurement is None:
            raise AFDInfeasible(
                AFDReasonCategory.INCOMPATIBLE_PHASE,
                f"AFD foreground engine has no {resolved_phase.value!r} measurement",
            )
        evaluation = evaluate_afd_phase(
            self._topology,
            measurement,
            input_length=input_length,
            output_length=output_length,
            latency_correction=latency_correction,
        )

        durations = (
            (AFDStage.ATTENTION, float(measurement.attention_ms)),
            (AFDStage.A_TO_F, float(measurement.a_to_f_ms) * self._topology.comm_overhead_factor),
            (AFDStage.FFN, float(measurement.ffn_ms)),
            (AFDStage.F_TO_A, float(measurement.f_to_a_ms) * self._topology.comm_overhead_factor),
        )
        intervals: list[AFDStageInterval] = []
        unit_index = 0
        for layer in range(measurement.num_layers):
            for microbatch in range(self._topology.num_microbatches):
                cursor = now + unit_index * evaluation.cycle_ms * latency_correction
                for stage, duration in durations:
                    end = cursor + duration * latency_correction
                    intervals.append(
                        AFDStageInterval(
                            stage=stage,
                            layer=layer,
                            microbatch=microbatch,
                            start_ms=cursor,
                            end_ms=end,
                        )
                    )
                    cursor = end
                unit_index += 1
        end_ms = now + evaluation.step_latency_ms
        if intervals and not math.isclose(intervals[-1].end_ms, end_ms, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("AFD foreground schedule drifted from the evaluated full-pass boundary")
        planned = AFDForegroundPass(
            pass_id=self._next_pass_id,
            phase=resolved_phase,
            started_at_ms=now,
            end_ms=end_ms,
            input_length=input_length,
            output_length=output_length,
            evaluation=evaluation,
            intervals=tuple(intervals),
        )
        self._next_pass_id += 1
        self._in_flight = planned
        return planned

    def complete_pass(self, pass_id: int, *, now_ms: float) -> AFDForegroundCompletion:
        """Release one pass at or after its modeled foreground boundary."""

        now = _validate_time(now_ms, "pass completion time")
        pending = self._in_flight
        if pending is None:
            raise RuntimeError("cannot complete an AFD foreground pass when none is in flight")
        if isinstance(pass_id, bool) or not isinstance(pass_id, int) or pass_id != pending.pass_id:
            raise ValueError(f"AFD foreground pass ID mismatch: expected {pending.pass_id}, got {pass_id!r}")
        if now < pending.end_ms:
            raise ValueError(
                f"AFD foreground pass {pass_id} completed at {now}ms before its modeled boundary {pending.end_ms}ms"
            )
        self._in_flight = None
        completed_sequences = self._topology.total_batch_size
        completed_tokens = (
            completed_sequences * pending.input_length if pending.phase is AFDPhase.PREFILL else completed_sequences
        )
        return AFDForegroundCompletion(
            pass_id=pass_id,
            phase=pending.phase,
            completed_at_ms=pending.end_ms,
            completed_sequences=completed_sequences,
            completed_tokens=completed_tokens,
            pass_latency_ms=pending.end_ms - pending.started_at_ms,
        )


__all__ = [
    "AFDForegroundCompletion",
    "AFDForegroundEngine",
    "AFDForegroundPass",
    "AFDPhaseEvaluation",
    "AFDStage",
    "AFDStageInterval",
    "evaluate_afd_phase",
]
