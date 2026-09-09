# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable replay and runner contracts owned by AI Simulate."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .provider import AdapterReplaySpec, JSONValue, RuntimeHookSpec

REPLAY_SPEC_API_VERSION = 1


@dataclass(frozen=True)
class EstimatorSpec:
    """Resolved, request-scoped estimator and performance-data identity.

    This contract is deliberately concrete: a runner never resolves ``latest``
    independently, consults process-global system paths, or guesses an empirical
    transfer policy after the search has begun.
    """

    model_path: str
    model_architecture: str
    system: str
    backend: str
    backend_version: str
    performance_data_version: str
    database_mode: str
    transfer_policy: tuple[str, ...]
    forward_model: str
    engine_step_backend: str
    systems_paths: tuple[str, ...]
    performance_data_root: str


@dataclass(frozen=True)
class RoleEngineRequestSpec:
    """Resolved engine controls for exactly one disaggregated role."""

    role: str
    backend: str
    backend_version: str
    cached_prefix_tokens: int = 0
    context_tokens: int = 0
    enable_chunked_prefill: bool = False
    enable_wideep: bool = False
    enable_eplb: bool = False
    wideep_num_slots: int | None = None
    moe_backend: str | None = None
    attention_backend: str | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    nextn: int = 0
    nextn_accepted: float | None = None
    memory_fraction_kind: str = "of_total"
    memory_fraction: float = 0.9
    max_seq_len: int = 0
    model_family: str = ""


@dataclass(frozen=True)
class EngineRequestSpec:
    """Resolved legacy engine/request controls evaluated for one candidate."""

    cached_prefix_tokens: int = 0
    context_tokens: dict[str, int] = field(default_factory=dict)
    enable_chunked_prefill: bool = False
    enable_wideep: bool = False
    enable_eplb: bool = False
    wideep_num_slots: int | None = None
    moe_backend: str | None = None
    attention_backend: str | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    nextn: int = 0
    nextn_accepted: float | None = None
    memory_fraction_kind: str = "of_total"
    memory_fraction_by_role: dict[str, float] = field(default_factory=dict)
    max_seq_len: int = 0
    model_family: str = ""
    role_requests: dict[str, RoleEngineRequestSpec] = field(default_factory=dict)

    def for_role(self, role: str) -> RoleEngineRequestSpec | None:
        """Return the explicit role request for heterogeneous P/D, if present."""

        return self.role_requests.get(role)


@dataclass(frozen=True)
class DisaggregatedCorrectionSpec:
    """Legacy-calibrated role service corrections consumed by replay."""

    prefill_rate_degradation: float = 0.9
    decode_rate_degradation: float = 0.92
    prefill_latency_correction: float = 1.1
    decode_latency_correction: float = 1.08
    ttft_correction_factor: float = 1.8

    def __post_init__(self) -> None:
        for name in (
            "prefill_rate_degradation",
            "decode_rate_degradation",
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
                raise ValueError(
                    f"{name} must be a positive finite number, got {value!r}"
                )

    def latency_scale(self, role: str) -> float:
        """Lower role rate and latency corrections to one service-time scale."""

        if role == "prefill":
            scale = (
                self.prefill_latency_correction
                * self.ttft_correction_factor
                / self.prefill_rate_degradation
            )
        elif role == "decode":
            scale = self.decode_latency_correction / self.decode_rate_degradation
        else:
            raise ValueError(
                f"correction role must be 'prefill' or 'decode', got {role!r}"
            )
        if not math.isfinite(scale):
            raise ValueError(f"{role} correction scale overflowed to {scale!r}")
        return scale


@dataclass(frozen=True)
class EncoderWorkerSpec:
    """One concrete encoder-only worker pool for an EPD candidate."""

    candidate_id: str
    backend_key: str
    estimator: EstimatorSpec
    tp: int
    batch_size: int
    num_workers: int
    latency_ms: float
    throughput_rps_per_worker: float
    memory_gib_per_worker: float
    rate_degradation: float
    power_w_per_worker: float
    power_coverage: float

    def __post_init__(self) -> None:
        for name in ("candidate_id", "backend_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("tp", "batch_size", "num_workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"encoder {name} must be a positive integer")
        for name in (
            "latency_ms",
            "throughput_rps_per_worker",
            "rate_degradation",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"encoder {name} must be positive and finite")
        for name in ("memory_gib_per_worker",):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"encoder {name} must be non-negative and finite")
        for name in ("power_w_per_worker", "power_coverage"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"encoder {name} must be positive and finite")
        if self.power_coverage > 1.0:
            raise ValueError("encoder power_coverage must be within [0, 1]")

    @property
    def total_gpus(self) -> int:
        return self.tp * self.num_workers

    @property
    def degraded_capacity_rps(self) -> float:
        return self.throughput_rps_per_worker * self.rate_degradation * self.num_workers


@dataclass(frozen=True)
class EpdDeploymentSpec:
    """Encoder overlay applied to an aggregate or P/D language deployment."""

    encoder: EncoderWorkerSpec
    language_gpus: int
    language_topology: str = "agg"
    ttft_scale: float = 1.0
    artifact_generation_supported: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.language_gpus, bool) or not isinstance(self.language_gpus, int) or self.language_gpus < 1:
            raise ValueError("EPD language_gpus must be a positive integer")
        if self.language_topology not in {"agg", "disagg"}:
            raise ValueError("EPD language_topology must be 'agg' or 'disagg'")
        if (
            isinstance(self.ttft_scale, bool)
            or not isinstance(self.ttft_scale, (int, float))
            or not math.isfinite(float(self.ttft_scale))
            or float(self.ttft_scale) <= 0.0
        ):
            raise ValueError("EPD ttft_scale must be positive and finite")
        if self.artifact_generation_supported:
            raise ValueError("EPD deployment artifact generation is not supported by native Sweeper")

    @property
    def total_gpus(self) -> int:
        return self.language_gpus + self.encoder.total_gpus


@dataclass(frozen=True)
class BackendDeploymentSpec:
    """Concrete backend engines and fleet shape for one candidate."""

    deployment_mode: str
    backend: str
    backend_version: str
    parallel_config: dict[str, JSONValue] = field(default_factory=dict)
    agg_engine_args: dict[str, JSONValue] | None = None
    prefill_engine_args: dict[str, JSONValue] | None = None
    decode_engine_args: dict[str, JSONValue] | None = None
    num_workers: int = 0
    num_prefill_workers: int = 0
    num_decode_workers: int = 0
    # Appended to preserve the positional constructor slots above.
    estimator: EstimatorSpec | None = None
    engine_request: EngineRequestSpec | None = None
    prefill_backend: str | None = None
    prefill_backend_version: str | None = None
    decode_backend: str | None = None
    decode_backend_version: str | None = None
    role_estimators: dict[str, EstimatorSpec] = field(default_factory=dict)
    disaggregated_corrections: DisaggregatedCorrectionSpec | None = None
    epd: EpdDeploymentSpec | None = None


@dataclass(frozen=True)
class ReplaySpec:
    """Strict data boundary between Sweeper and an injected replay runner."""

    backend_deployment: BackendDeploymentSpec
    workload: dict[str, JSONValue]
    goal: dict[str, JSONValue]
    concurrency: int | None = None
    adapters: dict[str, AdapterReplaySpec] = field(default_factory=dict)
    api_version: int = REPLAY_SPEC_API_VERSION

    @property
    def runtime_hooks(self) -> tuple[RuntimeHookSpec, ...]:
        """All requested hooks in deterministic adapter insertion order."""

        return tuple(
            hook
            for adapter_spec in self.adapters.values()
            for hook in adapter_spec.runtime_hooks
        )


@dataclass(frozen=True)
class ReplayReport:
    """Runner output consumed by Sweeper scoring."""

    metrics: dict[str, float]
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayOutputRequirements:
    """Optional detail requested from a Runner without changing replay semantics."""

    include_raw_report: bool = False
    capture_per_request: bool = False


@dataclass(frozen=True, order=True)
class HookCapability:
    """One runtime-hook ABI supported by a runner composition."""

    provider: str
    kind: str
    api_version: int

    def supports(self, hook: RuntimeHookSpec) -> bool:
        return (
            self.provider == hook.provider
            and self.kind == hook.kind
            and type(self.api_version) is int
            and type(hook.api_version) is int
            and self.api_version == hook.api_version
        )


@dataclass(frozen=True)
class RunnerCapabilities:
    """Replay-spec, backend/topology, and runtime-hook support advertised up front."""

    replay_spec_api_version: int = REPLAY_SPEC_API_VERSION
    supported_backend_topologies: tuple[tuple[str, str], ...] = ()
    supported_hooks: tuple[HookCapability, ...] = ()
    supports_disaggregated_attention_dp: bool = False
    supported_disaggregated_backend_pairs: tuple[tuple[str, str], ...] = ()
    supported_epd_backend_topologies: tuple[tuple[str, str], ...] = ()

    def supports_backend_topology(self, backend: str, topology: str) -> bool:
        """Return whether a backend/topology pair is supported.

        ``"*"`` may be used in either position by a runner that supports a
        complete backend or topology family.
        """

        return any(
            (supported_backend in (backend, "*"))
            and (supported_topology in (topology, "*"))
            for supported_backend, supported_topology in self.supported_backend_topologies
        )

    def supports_hook(self, hook: RuntimeHookSpec) -> bool:
        return any(capability.supports(hook) for capability in self.supported_hooks)

    def supports_epd(self, backend: str, topology: str) -> bool:
        """Return whether the runner explicitly supports an encoder overlay."""

        return any(
            supported_backend in (backend, "*") and supported_topology in (topology, "*")
            for supported_backend, supported_topology in self.supported_epd_backend_topologies
        )

    def supports_disaggregated_backend_pair(
        self, prefill_backend: str, decode_backend: str
    ) -> bool:
        """Return whether a runner explicitly supports one P/D backend pair."""

        return any(
            supported_prefill in (prefill_backend, "*")
            and supported_decode in (decode_backend, "*")
            for supported_prefill, supported_decode in self.supported_disaggregated_backend_pairs
        )

    def supports_attention_dp(self, topology: str, *dp_sizes: int) -> bool:
        """Return whether the topology supports all requested attention-DP sizes."""

        return (
            topology != "disagg"
            or self.supports_disaggregated_attention_dp
            or all(dp_size == 1 for dp_size in dp_sizes)
        )

    def require_replay_spec_version(
        self, api_version: int = REPLAY_SPEC_API_VERSION
    ) -> None:
        """Raise when the runner and Sweeper do not share the replay-spec ABI."""

        versions_are_integers = (
            type(api_version) is int and type(self.replay_spec_api_version) is int
        )
        if not versions_are_integers or api_version != self.replay_spec_api_version:
            raise ValueError(
                f"ReplaySpec API version {api_version} is incompatible with "
                f"runner version {self.replay_spec_api_version}"
            )

    def require_compatible(self, spec: ReplaySpec) -> None:
        """Raise a clear error when this runner cannot execute ``spec``."""

        self.require_replay_spec_version(spec.api_version)
        deployment = spec.backend_deployment
        if deployment.epd is not None:
            encoder_backend = deployment.epd.encoder.estimator.backend
            if not self.supports_epd(encoder_backend, deployment.deployment_mode):
                raise ValueError(
                    "runner does not support EPD encoder backend/topology "
                    f"{encoder_backend!r}/{deployment.deployment_mode!r}"
                )
        if deployment.deployment_mode == "disagg" and (
            deployment.prefill_backend is not None
            or deployment.decode_backend is not None
        ):
            role_backends = {
                "prefill": deployment.prefill_backend,
                "decode": deployment.decode_backend,
            }
            for role, backend in role_backends.items():
                if backend is None or not self.supports_backend_topology(
                    backend, deployment.deployment_mode
                ):
                    raise ValueError(
                        f"runner does not support {role} backend/topology {backend!r}/{deployment.deployment_mode!r}"
                    )
            assert deployment.prefill_backend is not None
            assert deployment.decode_backend is not None
            if not self.supports_disaggregated_backend_pair(
                deployment.prefill_backend, deployment.decode_backend
            ):
                raise ValueError(
                    "runner does not explicitly support disaggregated backend pair "
                    f"prefill={deployment.prefill_backend!r}, "
                    f"decode={deployment.decode_backend!r}"
                )
        elif not self.supports_backend_topology(
            deployment.backend, deployment.deployment_mode
        ):
            raise ValueError(
                f"runner does not support backend/topology {deployment.backend!r}/{deployment.deployment_mode!r}"
            )
        unsupported = [
            hook for hook in spec.runtime_hooks if not self.supports_hook(hook)
        ]
        if unsupported:
            labels = ", ".join(
                f"{hook.provider}:{hook.kind}@{hook.api_version}"
                for hook in unsupported
            )
            raise ValueError(f"runner does not support runtime hook(s): {labels}")


@runtime_checkable
class Runner(Protocol):
    """One worker-local replay executor."""

    def run(
        self,
        spec: ReplaySpec,
        *,
        output_requirements: ReplayOutputRequirements | None = None,
    ) -> ReplayReport: ...

    def close(self) -> None: ...


@runtime_checkable
class RunnerFactory(Protocol):
    """Serializable factory used to create one reusable Runner per worker."""

    def capabilities(self) -> RunnerCapabilities: ...

    def create(self, worker_id: int) -> Runner: ...


def _jsonable(value: Any) -> JSONValue:
    """Recursively convert supported contract values into JSON data."""

    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"canonical replay JSON requires string mapping keys, got {key!r}"
                )
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"value of type {type(value).__name__} is not supported by replay JSON contracts"
    )


def validate_json_value(value: Any, *, path: str = "value") -> None:
    """Require an exact JSON value without silently normalizing Python objects.

    ``canonical_json`` accepts the Sweeper contract dataclasses themselves and
    converts them to JSON for cache keys and diagnostics. Adapter-owned payloads,
    however, cross a process/package ABI and must already consist only of JSON
    primitives, lists, and string-keyed dictionaries.
    """

    value_type = type(value)
    if value is None or value_type in (str, int, bool):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if value_type is list:
        for index, item in enumerate(value):
            validate_json_value(item, path=f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} requires string mapping keys, got {key!r}")
            validate_json_value(item, path=f"{path}[{key!r}]")
        return
    raise TypeError(f"{path} contains non-JSON value of type {value_type.__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic, strict JSON suitable for serialization and cache keys."""

    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
