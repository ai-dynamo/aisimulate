# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral contracts for heterogeneous prefill/decode deployments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .replay import EstimatorSpec


class DisaggRole(str, Enum):
    """A role in a prefill/decode disaggregated deployment."""

    PREFILL = "prefill"
    DECODE = "decode"


class RoleFailureCategory(str, Enum):
    """Stable failure categories for role-specific search diagnostics."""

    INVALID_IDENTITY = "invalid_identity"
    ESTIMATOR_RESOLUTION = "estimator_resolution"
    ENGINE_CONTROLS = "engine_controls"
    KV_CAPACITY = "kv_capacity"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    NO_PARALLEL_CONFIG = "no_parallel_config"
    GPU_BUDGET = "gpu_budget"
    INVALID_ESTIMATE = "invalid_estimate"


class RoleSearchError(ValueError):
    """A role-attributed disaggregated-search failure."""

    def __init__(
        self,
        role: DisaggRole | str,
        category: RoleFailureCategory,
        detail: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.role = DisaggRole(role)
        self.category = category
        self.detail = detail
        self.provenance = dict(provenance or {})
        super().__init__(f"{self.role.value} {category.value}: {detail}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "category": self.category.value,
            "detail": self.detail,
            "provenance": dict(self.provenance),
        }


def _nonempty(name: str, value: str, *, role: DisaggRole) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RoleSearchError(
            role,
            RoleFailureCategory.INVALID_IDENTITY,
            f"{name} must be a non-empty string, got {value!r}",
            provenance={"field": name, "value": value},
        )


def _positive_int(name: str, value: int, *, role: DisaggRole) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RoleSearchError(
            role,
            RoleFailureCategory.INVALID_ESTIMATE,
            f"{name} must be a positive integer, got {value!r}",
            provenance={"field": name, "value": value},
        )


def _positive_finite(name: str, value: float, *, role: DisaggRole) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RoleSearchError(
            role,
            RoleFailureCategory.INVALID_ESTIMATE,
            f"{name} must be a positive finite number, got {value!r}",
            provenance={"field": name, "value": value},
        )


@dataclass(frozen=True, order=True)
class DisaggBackendPair:
    """One searched prefill/decode backend combination."""

    prefill: str
    decode: str

    def __post_init__(self) -> None:
        _nonempty("backend", self.prefill, role=DisaggRole.PREFILL)
        _nonempty("backend", self.decode, role=DisaggRole.DECODE)
        object.__setattr__(self, "prefill", self.prefill.strip())
        object.__setattr__(self, "decode", self.decode.strip())

    @property
    def homogeneous(self) -> bool:
        return self.prefill == self.decode

    @property
    def label(self) -> str:
        return f"prefill={self.prefill},decode={self.decode}"

    def backend_for(self, role: DisaggRole | str) -> str:
        resolved = DisaggRole(role)
        return self.prefill if resolved is DisaggRole.PREFILL else self.decode


@dataclass(frozen=True)
class RoleIdentity:
    """Resolved model, system, backend, and data version for one role."""

    role: DisaggRole | str
    model_name: str
    hardware_sku: str
    backend: str
    backend_version: str
    inherited_fields: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = DisaggRole(self.role)
        object.__setattr__(self, "role", role)
        for name in ("model_name", "hardware_sku", "backend", "backend_version"):
            _nonempty(name, getattr(self, name), role=role)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "model_name": self.model_name,
            "hardware_sku": self.hardware_sku,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "inherited_fields": list(self.inherited_fields),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class RoleEstimatorSpecs:
    """Immutable estimator/data identities for one searched backend pair."""

    pair: DisaggBackendPair
    prefill: EstimatorSpec
    decode: EstimatorSpec
    identities: Mapping[str, RoleIdentity]

    def __post_init__(self) -> None:
        if self.prefill.backend != self.pair.prefill:
            raise RoleSearchError(
                DisaggRole.PREFILL,
                RoleFailureCategory.ESTIMATOR_RESOLUTION,
                "prefill estimator backend does not match the searched backend pair",
            )
        if self.decode.backend != self.pair.decode:
            raise RoleSearchError(
                DisaggRole.DECODE,
                RoleFailureCategory.ESTIMATOR_RESOLUTION,
                "decode estimator backend does not match the searched backend pair",
            )
        expected = {DisaggRole.PREFILL.value, DisaggRole.DECODE.value}
        if set(self.identities) != expected:
            raise ValueError(f"identities must contain exactly {sorted(expected)}")
        for role in (DisaggRole.PREFILL, DisaggRole.DECODE):
            identity = self.identities[role.value]
            estimator = self.estimator_for(role)
            expected_identity = (
                role,
                estimator.model_path,
                estimator.system,
                estimator.backend,
                estimator.backend_version,
            )
            actual_identity = (
                identity.role,
                identity.model_name,
                identity.hardware_sku,
                identity.backend,
                identity.backend_version,
            )
            if actual_identity != expected_identity:
                raise RoleSearchError(
                    role,
                    RoleFailureCategory.INVALID_IDENTITY,
                    "role identity does not match its resolved estimator",
                    provenance={
                        "identity": identity.as_dict(),
                        "estimator": {
                            "model_name": estimator.model_path,
                            "hardware_sku": estimator.system,
                            "backend": estimator.backend,
                            "backend_version": estimator.backend_version,
                        },
                    },
                )
        if self.prefill.model_architecture != self.decode.model_architecture:
            raise RoleSearchError(
                DisaggRole.DECODE,
                RoleFailureCategory.INVALID_IDENTITY,
                "prefill/decode model architectures are incompatible for KV handoff: "
                f"{self.prefill.model_architecture!r} != "
                f"{self.decode.model_architecture!r}",
                provenance={
                    "prefill_model": self.prefill.model_path,
                    "prefill_architecture": self.prefill.model_architecture,
                    "decode_model": self.decode.model_path,
                    "decode_architecture": self.decode.model_architecture,
                },
            )
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))

    def estimator_for(self, role: DisaggRole | str) -> EstimatorSpec:
        resolved = DisaggRole(role)
        return self.prefill if resolved is DisaggRole.PREFILL else self.decode

    def identity_for(self, role: DisaggRole | str) -> RoleIdentity:
        return self.identities[DisaggRole(role).value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_pair": self.pair.label,
            "identities": {role: identity.as_dict() for role, identity in self.identities.items()},
        }


@dataclass(frozen=True)
class DisaggRateMatchControls:
    """Legacy-calibrated disaggregated rate and latency corrections."""

    prefill_degradation: float = 0.9
    decode_degradation: float = 0.92
    prefill_latency_correction: float = 1.1
    decode_latency_correction: float = 1.08
    ttft_correction_factor: float = 1.8

    def __post_init__(self) -> None:
        for name in (
            "prefill_degradation",
            "decode_degradation",
            "prefill_latency_correction",
            "decode_latency_correction",
            "ttft_correction_factor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")


@dataclass(frozen=True)
class RoleEstimate:
    """One role's standalone estimate before P/D rate matching."""

    identity: RoleIdentity
    sequence_rate_per_worker: float
    latency_ms: float
    workers: int
    gpus_per_worker: int
    parallel_config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = self.identity.role
        _positive_finite("sequence_rate_per_worker", self.sequence_rate_per_worker, role=role)
        _positive_finite("latency_ms", self.latency_ms, role=role)
        _positive_int("workers", self.workers, role=role)
        _positive_int("gpus_per_worker", self.gpus_per_worker, role=role)
        object.__setattr__(self, "parallel_config", MappingProxyType(dict(self.parallel_config)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def total_gpus(self) -> int:
        return self.workers * self.gpus_per_worker

    @property
    def standalone_sequence_rate(self) -> float:
        value = self.workers * self.sequence_rate_per_worker
        _positive_finite("standalone_sequence_rate", value, role=self.identity.role)
        return value


@dataclass(frozen=True)
class DisaggRateMatchResult:
    """Rate-matched end-to-end result with role-lossless accounting."""

    sequence_rate: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    total_gpus: int
    prefill_gpus: int
    decode_gpus: int
    limiting_role: DisaggRole
    role_rates: Mapping[str, float]
    role_identities: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "sequence_rate",
            "tokens_per_second",
            "tokens_per_second_per_gpu",
            "ttft_ms",
            "tpot_ms",
            "request_latency_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"DisaggRateMatchResult.{name} must be finite and non-negative, got {value!r}"
                )
        for name, value in self.role_rates.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    "DisaggRateMatchResult.role_rates"
                    f"[{name!r}] must be finite and non-negative, got {value!r}"
                )
        object.__setattr__(self, "role_rates", MappingProxyType(dict(self.role_rates)))
        object.__setattr__(self, "role_identities", MappingProxyType(dict(self.role_identities)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_rate": self.sequence_rate,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_second_per_gpu": self.tokens_per_second_per_gpu,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "request_latency_ms": self.request_latency_ms,
            "total_gpus": self.total_gpus,
            "prefill_gpus": self.prefill_gpus,
            "decode_gpus": self.decode_gpus,
            "limiting_role": self.limiting_role.value,
            "role_rates": dict(self.role_rates),
            "role_identities": {role: dict(identity) for role, identity in self.role_identities.items()},
            "provenance": dict(self.provenance),
        }


def rate_match_disaggregated(
    prefill: RoleEstimate,
    decode: RoleEstimate,
    *,
    output_length: int,
    controls: DisaggRateMatchControls | None = None,
    gpu_budget: int | None = None,
) -> DisaggRateMatchResult:
    """Apply legacy P/D degradation, corrections, and GPU accounting."""

    if prefill.identity.role is not DisaggRole.PREFILL:
        raise RoleSearchError(
            prefill.identity.role,
            RoleFailureCategory.INVALID_ESTIMATE,
            "prefill estimate must carry role='prefill'",
        )
    if decode.identity.role is not DisaggRole.DECODE:
        raise RoleSearchError(
            decode.identity.role,
            RoleFailureCategory.INVALID_ESTIMATE,
            "decode estimate must carry role='decode'",
        )
    _positive_int("output_length", output_length, role=DisaggRole.DECODE)
    if gpu_budget is not None:
        _positive_int("gpu_budget", gpu_budget, role=DisaggRole.DECODE)

    resolved = controls or DisaggRateMatchControls()
    prefill_standalone_rate = prefill.standalone_sequence_rate
    decode_standalone_rate = decode.standalone_sequence_rate
    prefill_rate = prefill_standalone_rate * resolved.prefill_degradation
    decode_rate = decode_standalone_rate * resolved.decode_degradation
    _positive_finite(
        "effective_sequence_rate", prefill_rate, role=DisaggRole.PREFILL
    )
    _positive_finite(
        "effective_sequence_rate", decode_rate, role=DisaggRole.DECODE
    )
    sequence_rate = min(prefill_rate, decode_rate)
    limiting_role = DisaggRole.PREFILL if prefill_rate <= decode_rate else DisaggRole.DECODE
    prefill_gpus = prefill.total_gpus
    decode_gpus = decode.total_gpus
    total_gpus = prefill_gpus + decode_gpus
    if gpu_budget is not None and total_gpus > gpu_budget:
        raise RoleSearchError(
            limiting_role,
            RoleFailureCategory.GPU_BUDGET,
            f"heterogeneous disaggregated candidate uses {total_gpus} GPUs, exceeding gpu_budget={gpu_budget}",
            provenance={
                "prefill_gpus": prefill_gpus,
                "decode_gpus": decode_gpus,
                "total_gpus": total_gpus,
                "gpu_budget": gpu_budget,
            },
        )

    ttft_ms = prefill.latency_ms * resolved.prefill_latency_correction * resolved.ttft_correction_factor
    tpot_ms = decode.latency_ms * resolved.decode_latency_correction
    request_latency_ms = ttft_ms + tpot_ms * max(output_length - 1, 0)
    tokens_per_second = sequence_rate * output_length
    role_rates = {
        "prefill_standalone": prefill_standalone_rate,
        "prefill_effective": prefill_rate,
        "decode_standalone": decode_standalone_rate,
        "decode_effective": decode_rate,
    }
    return DisaggRateMatchResult(
        sequence_rate=sequence_rate,
        tokens_per_second=tokens_per_second,
        tokens_per_second_per_gpu=tokens_per_second / total_gpus,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        request_latency_ms=request_latency_ms,
        total_gpus=total_gpus,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        limiting_role=limiting_role,
        role_rates=role_rates,
        role_identities={
            DisaggRole.PREFILL.value: prefill.identity.as_dict(),
            DisaggRole.DECODE.value: decode.identity.as_dict(),
        },
        provenance={
            "formula": "min(prefill_workers * prefill_seq_s * degradation, "
            "decode_workers * decode_seq_s * degradation)",
            "controls": {
                "prefill_degradation": resolved.prefill_degradation,
                "decode_degradation": resolved.decode_degradation,
                "prefill_latency_correction": resolved.prefill_latency_correction,
                "decode_latency_correction": resolved.decode_latency_correction,
                "ttft_correction_factor": resolved.ttft_correction_factor,
            },
            "role_provenance": {
                DisaggRole.PREFILL.value: dict(prefill.provenance),
                DisaggRole.DECODE.value: dict(decode.provenance),
            },
        },
    )


__all__ = [
    "DisaggBackendPair",
    "DisaggRateMatchControls",
    "DisaggRateMatchResult",
    "DisaggRole",
    "RoleEstimate",
    "RoleEstimatorSpecs",
    "RoleFailureCategory",
    "RoleIdentity",
    "RoleSearchError",
    "rate_match_disaggregated",
]
