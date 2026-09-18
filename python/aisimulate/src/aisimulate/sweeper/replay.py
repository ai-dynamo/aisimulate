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

from ..config.common import ENGINE_MODEL_CONTROL_FIELDS, is_active_engine_model_control
from ..power import POWER_FIELDS, normalize_power_summary
from .provider import AdapterReplaySpec, JSONValue, RuntimeHookSpec

REPLAY_SPEC_API_VERSION = 1


@dataclass(frozen=True)
class ForwardPassEstimatorSpec:
    """Resolved output of Core's canonical forward-pass constructor.

    The config is the sole estimator identity carried by Sweeper and Replay.
    Convenience properties below are projections, never independently authored
    values. Diagnostics preserve Core's selection result for artifacts.
    """

    config: dict[str, JSONValue]
    options: dict[str, JSONValue] | None = None
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)

    @property
    def model_path(self) -> str:
        return str(self.config["model"])

    @property
    def system(self) -> str:
        return str(self.config["system"])

    @property
    def backend(self) -> str:
        return str(self.config["backend"])

    @property
    def backend_version(self) -> str:
        return str(self.config["backend_version"])

    @property
    def database_mode(self) -> str:
        return str(self.config["database_mode"])

    @property
    def transfer_policy(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config.get("transfer_policy") or ())

    @property
    def forward_model(self) -> str:
        return str(self.config["estimation_mode"])

    @property
    def systems_paths(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.config.get("systems_paths") or ())

    @property
    def performance_data_root(self) -> str:
        provenance = self.diagnostics.get("provenance")
        if isinstance(provenance, dict):
            return str(provenance.get("selected_systems_root") or "")
        return ""


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
    image_height: int
    image_width: int
    image_count: int
    power_w: float | None = None
    power_coverage: float = 0.0
    latency_correction: float = 1.0

    def __post_init__(self):
        for name in ("model", "system", "backend", "backend_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"encoder {name} must be nonempty")
        for name in ("tp", "batch_size", "workers", "visual_tokens", "image_height", "image_width", "image_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"encoder {name} must be a positive integer")
        for name in ("latency_ms", "throughput_rps", "memory_gib", "rate_degradation", "latency_correction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"encoder {name} must be positive and finite")
        if self.batch_size > 8 or self.rate_degradation > 1:
            raise ValueError("encoder batch_size must be <= 8 and rate_degradation <= 1")
        if (
            isinstance(self.power_coverage, bool)
            or not isinstance(self.power_coverage, Real)
            or not math.isfinite(self.power_coverage)
            or not 0 <= self.power_coverage <= 1
        ):
            raise ValueError("encoder power_coverage must be within [0, 1]")
        if self.power_w is not None and (
            isinstance(self.power_w, bool)
            or not isinstance(self.power_w, Real)
            or not math.isfinite(self.power_w)
            or self.power_w <= 0
            or self.power_coverage <= 0
        ):
            raise ValueError("encoder power requires positive finite watts and coverage")
        if self.power_w is None and self.power_coverage != 0:
            raise ValueError("unavailable encoder power must have zero coverage")

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
    forward_pass_estimators: dict[str, ForwardPassEstimatorSpec] = field(default_factory=dict)
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

    metrics: dict[str, float | None]
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        power = normalize_power_summary(self.metrics)
        for name, value in self.metrics.items():
            if name in POWER_FIELDS:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"runner metric {name} must be numeric; only power fields may be null")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"runner metric {name} must be finite") from exc
            if not math.isfinite(number):
                raise ValueError(f"runner metric {name} must be finite")
        object.__setattr__(self, "metrics", {**self.metrics, **power})


@dataclass(frozen=True)
class ReplayOutputRequirements:
    """Optional detail requested from a Runner without changing replay semantics."""

    include_raw_report: bool = False
    capture_per_request: bool = False
    capture_telemetry: bool = False
    telemetry_sample_interval_ms: float = 1000.0
    capture_memory_diagnostics: bool = False
    capture_performance_diagnostics: bool = False

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
    supported_trace_formats: tuple[str, ...] = ("*",)
    supports_agentic_lanes: bool = False
    supported_agentic_topologies: tuple[str, ...] = ("agg", "disagg")
    agentic_qualification: str | None = None
    supports_analytical_epd: bool = False
    supported_agentic_backends: tuple[str, ...] = ("*",)
    supports_agentic_host_offload: bool = True
    supports_agentic_speculative_decoding: bool = True
    supports_agentic_snapshots: bool = False
    supports_cached_prefix_tokens: bool = False
    supported_engine_model_controls: tuple[str, ...] = ()
    supports_mtp_expected_acceptance: bool = False
    supports_state_cache: bool = False

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

    def supports_trace_format(self, trace_format: str) -> bool:
        return "*" in self.supported_trace_formats or trace_format in self.supported_trace_formats

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
        is_afd = deployment.deployment_mode in {"afd", "afd+pd"}
        if spec.workload.get("cached_prefix_tokens"):
            if is_afd:
                raise ValueError("cached_prefix_tokens is unsupported for AFD")
            if not self.supports_cached_prefix_tokens:
                raise ValueError("runner does not support cached_prefix_tokens; use --stack engine")
        for args in (deployment.agg_engine_args, deployment.prefill_engine_args, deployment.decode_engine_args):
            if not isinstance(args, Mapping):
                continue
            rank = args.get("rank", args)
            if not isinstance(rank, Mapping):
                continue
            timing = rank.get("timing_model")
            identity = timing.get("config", {}) if isinstance(timing, Mapping) else {}
            if not isinstance(identity, Mapping):
                # The timing validator handles malformed identities; still inspect flat controls.
                identity = {}
            active_controls = [
                name
                for name in ENGINE_MODEL_CONTROL_FIELDS
                if any(
                    is_active_engine_model_control(name, value)
                    for value in (identity.get(name), rank.get(name), rank.get(f"aic_{name}"))
                )
            ]
            if is_afd and active_controls:
                raise ValueError(f"engine model controls {active_controls} are unsupported for AFD")
            unsupported_controls = [
                name for name in active_controls if name not in self.supported_engine_model_controls
            ]
            if unsupported_controls:
                raise ValueError(
                    f"runner does not support engine model controls {unsupported_controls}; use --stack engine"
                )
            if any(rank.get(name) is not None for name in ("aic_nextn_accepted", "nextn_accepted")):
                if is_afd:
                    raise ValueError("explicit MTP expected acceptance is unsupported for AFD")
                if not self.supports_mtp_expected_acceptance:
                    raise ValueError("runner does not support explicit MTP expected acceptance; use --stack engine")
            if is_afd and rank.get("enable_chunked_prefill") is not None:
                raise ValueError("enable_chunked_prefill is unsupported for AFD")
        if not self.supports_state_cache:
            for args in (deployment.agg_engine_args, deployment.prefill_engine_args, deployment.decode_engine_args):
                if not args:
                    continue
                rank = args.get("rank", args)
                if isinstance(rank, Mapping) and rank.get("state_cache") is not None:
                    raise ValueError(
                        "runner does not support state_cache; select a stack that advertises this capability"
                    )
        if deployment.encoder is not None and deployment.deployment_mode not in {"agg", "disagg"}:
            raise ValueError("analytical EPD supports only agg/disagg language deployments; AFD is unsupported")
        if deployment.encoder is not None and not self.supports_analytical_epd:
            raise ValueError("runner does not support analytical EPD")
        if not self.supports_backend_topology(deployment.backend, deployment.deployment_mode):
            raise ValueError(
                f"runner does not support backend/topology {deployment.backend!r}/{deployment.deployment_mode!r}"
            )
        trace_format = spec.workload.get("trace_format")
        if isinstance(trace_format, str) and not self.supports_trace_format(trace_format):
            raise ValueError(f"runner does not support trace format {trace_format!r}")
        weka_basis = spec.workload.get("weka_nested_timestamp_basis")
        if weka_basis is not None:
            if trace_format != "weka":
                raise ValueError("weka_nested_timestamp_basis requires Weka input")
            if weka_basis not in {"auto", "absolute", "relative"}:
                raise ValueError("weka_nested_timestamp_basis must be 'auto', 'absolute', or 'relative'")
        agentic_lanes = spec.workload.get("agentic_lanes")
        if agentic_lanes is not None:
            if type(agentic_lanes) is not int or agentic_lanes <= 0:
                raise ValueError("agentic_lanes must be a positive integer")
            if trace_format not in {"weka", "agentic_mooncake", "dynamo"}:
                raise ValueError("agentic_lanes requires weka, agentic_mooncake, or agentic dynamo input")
            if not self.supports_agentic_lanes:
                raise ValueError("runner does not support agentic_lanes")
        agentic_snapshot = spec.workload.get("agentic_snapshot")
        if agentic_snapshot is not None:
            if (
                not isinstance(agentic_snapshot, Mapping)
                or set(agentic_snapshot) != {"seed"}
                or type(agentic_snapshot["seed"]) is not int
                or not 0 <= agentic_snapshot["seed"] <= 0xFFFF_FFFF_FFFF_FFFF
            ):
                raise ValueError("agentic_snapshot requires exactly one unsigned 64-bit integer seed")
            if agentic_lanes is None or spec.workload.get("load_type") != "trace_timestamps":
                raise ValueError("agentic_snapshot requires trace_timestamps load with positive agentic_lanes")
            if spec.workload.get("source_type") != "trace" or spec.workload.get("replay_concurrency") is not None:
                raise ValueError("agentic_snapshot requires agentic trace input without replay_concurrency")
            if not self.supports_agentic_snapshots:
                raise ValueError("runner does not support agentic snapshots")
        agentic_topology_required = trace_format in {"weka", "agentic_mooncake"} or (
            trace_format == "dynamo" and agentic_lanes is not None
        )
        if agentic_topology_required and deployment.deployment_mode not in self.supported_agentic_topologies:
            raise ValueError(
                f"runner does not support agentic trace format {trace_format!r} "
                f"with topology {deployment.deployment_mode!r}"
            )
        if agentic_topology_required:
            if "*" not in self.supported_agentic_backends and deployment.backend not in self.supported_agentic_backends:
                raise ValueError(f"runner does not support agentic execution with backend {deployment.backend!r}")
            role_args = (
                [deployment.agg_engine_args]
                if deployment.deployment_mode == "agg"
                else [deployment.prefill_engine_args, deployment.decode_engine_args]
            )
            for args in role_args:
                if not args:
                    continue
                rank = args.get("rank", args)
                if not isinstance(rank, Mapping):
                    continue  # The engine descriptor validator reports malformed ranks.
                if not self.supports_agentic_host_offload and rank.get("native_host_offload") is not None:
                    raise ValueError("agentic replay requires HBM-only KV cache; host offload is unsupported")
                if not self.supports_agentic_speculative_decoding and any(
                    rank.get(key) is not None for key in ("aic_nextn", "nextn", "speculation")
                ):
                    raise ValueError("agentic replay requires speculative decoding disabled")
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
