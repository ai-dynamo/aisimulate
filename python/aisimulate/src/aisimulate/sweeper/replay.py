# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable replay and runner contracts owned by AI Simulate."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .provider import AdapterReplaySpec, JSONValue, RuntimeHookSpec

REPLAY_SPEC_API_VERSION = 1


@dataclass(frozen=True)
class EncoderPoolSpec:
    """Resolved analytical EPD pool. Missing power is unavailable, never zero watts."""

    model: str
    system: str
    backend: str
    backend_version: str
    tp: int
    batch_size: int
    workers: int
    latency_ms: float
    throughput_rps: float
    memory_gib: float
    rate_degradation: float
    visual_tokens: int
    power_w: float | None = None
    power_coverage: float = 0.0

    def __post_init__(self):
        for name in ("model", "system", "backend", "backend_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"encoder {name} must be nonempty")
        for name in ("tp", "batch_size", "workers", "visual_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"encoder {name} must be a positive integer")
        for name in ("latency_ms", "throughput_rps", "memory_gib", "rate_degradation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"encoder {name} must be positive and finite")
        if self.batch_size > 8 or self.rate_degradation > 1:
            raise ValueError("encoder batch_size must be <= 8 and rate_degradation <= 1")
        if not math.isfinite(self.power_coverage) or not 0 <= self.power_coverage <= 1:
            raise ValueError("encoder power_coverage must be within [0, 1]")
        if self.power_w is not None and (
            not math.isfinite(self.power_w) or self.power_w <= 0 or self.power_coverage <= 0
        ):
            raise ValueError("encoder power requires positive finite watts and coverage")

    @property
    def total_gpus(self) -> int:
        return self.tp * self.workers


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
    performance_model_metadata: dict[str, JSONValue] = field(default_factory=dict)
    encoder: EncoderPoolSpec | None = None


@dataclass(frozen=True)
class ReplaySpec:
    """Strict data boundary between Sweeper and an injected replay runner."""

    backend_deployment: BackendDeploymentSpec
    workload: dict[str, JSONValue]
    goal: dict[str, JSONValue]
    concurrency: int | None = None
    adapters: dict[str, AdapterReplaySpec] = field(default_factory=dict)
    api_version: int = REPLAY_SPEC_API_VERSION
    execution_mode: str = "offline"

    @property
    def runtime_hooks(self) -> tuple[RuntimeHookSpec, ...]:
        """All requested hooks in deterministic adapter insertion order."""

        return tuple(hook for adapter_spec in self.adapters.values() for hook in adapter_spec.runtime_hooks)


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
    capture_telemetry: bool = False
    telemetry_sample_interval_ms: float = 1000.0

    def __post_init__(self) -> None:
        interval = self.telemetry_sample_interval_ms
        if self.capture_telemetry and (
            isinstance(interval, bool)
            or not isinstance(interval, Real)
            or not math.isfinite(interval)
            or interval <= 0.0
        ):
            raise ValueError(
                "telemetry_sample_interval_ms must be finite and positive when capture_telemetry is enabled"
            )


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
    supported_execution_modes: tuple[str, ...] = ("offline",)
    supports_analytical_epd: bool = False

    def supports_backend_topology(self, backend: str, topology: str) -> bool:
        """Return whether a backend/topology pair is supported.

        ``"*"`` may be used in either position by a runner that supports a
        complete backend or topology family.
        """

        return any(
            (supported_backend in (backend, "*")) and (supported_topology in (topology, "*"))
            for supported_backend, supported_topology in self.supported_backend_topologies
        )

    def supports_execution_mode(self, mode: str) -> bool:
        """Return whether this runner can execute the requested clock mode."""

        return mode in self.supported_execution_modes

    def supports_hook(self, hook: RuntimeHookSpec) -> bool:
        return any(capability.supports(hook) for capability in self.supported_hooks)

    def supports_attention_dp(self, topology: str, *dp_sizes: int) -> bool:
        """Return whether the topology supports all requested attention-DP sizes."""

        return (
            topology != "disagg"
            or self.supports_disaggregated_attention_dp
            or all(dp_size == 1 for dp_size in dp_sizes)
        )

    def require_replay_spec_version(self, api_version: int = REPLAY_SPEC_API_VERSION) -> None:
        """Raise when the runner and Sweeper do not share the replay-spec ABI."""

        versions_are_integers = type(api_version) is int and type(self.replay_spec_api_version) is int
        if not versions_are_integers or api_version != self.replay_spec_api_version:
            raise ValueError(
                f"ReplaySpec API version {api_version} is incompatible with "
                f"runner version {self.replay_spec_api_version}"
            )

    def require_compatible(self, spec: ReplaySpec) -> None:
        """Raise a clear error when this runner cannot execute ``spec``."""

        self.require_replay_spec_version(spec.api_version)
        if not self.supports_execution_mode(spec.execution_mode):
            raise ValueError(f"runner does not support execution mode {spec.execution_mode!r}")
        deployment = spec.backend_deployment
        if deployment.encoder is not None and not self.supports_analytical_epd:
            raise ValueError("runner does not support analytical EPD")
        if not self.supports_backend_topology(deployment.backend, deployment.deployment_mode):
            raise ValueError(
                f"runner does not support backend/topology {deployment.backend!r}/{deployment.deployment_mode!r}"
            )
        unsupported = [hook for hook in spec.runtime_hooks if not self.supports_hook(hook)]
        if unsupported:
            labels = ", ".join(f"{hook.provider}:{hook.kind}@{hook.api_version}" for hook in unsupported)
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
                raise TypeError(f"canonical replay JSON requires string mapping keys, got {key!r}")
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not supported by replay JSON contracts")


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
