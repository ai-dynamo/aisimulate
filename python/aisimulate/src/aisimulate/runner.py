# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-free engine replay implementation of the Sweeper Runner contract."""

from __future__ import annotations

import heapq
import importlib
import json
import logging
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

from .capacity import materialize_aic_num_gpu_blocks
from .config.common import ENGINE_MODEL_CONTROL_FIELDS, is_active_engine_model_control
from .config.engine import StateCacheConfig
from .power import normalize_power_summary, power_metadata
from .sweeper.afd_engine import AFDForegroundEngine
from .sweeper.afd_parallel import AFDPhase, AFDTopology
from .sweeper.afd_perfmodel import AFDLayerTimes
from .sweeper.provider import JSONValue
from .sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)
from .traffic import materialize_configured_traffic

logger = logging.getLogger(__name__)

_SUPPORTED_BACKEND_TOPOLOGIES = (
    ("vllm", "agg"),
    ("vllm", "disagg"),
    ("sglang", "agg"),
    ("sglang", "disagg"),
    ("trtllm", "agg"),
    ("trtllm", "disagg"),
    ("vllm", "afd"),
    ("vllm", "afd+pd"),
    ("sglang", "afd"),
    ("sglang", "afd+pd"),
    ("trtllm", "afd"),
    ("trtllm", "afd+pd"),
)

_RUNTIME_TRAFFIC_FIELDS = frozenset(
    {
        "source_type",
        "load_type",
        "trace_path",
        "trace_paths",
        "trace_format",
        "trace_block_size",
        "weka_nested_timestamp_basis",
        "arrival_speedup_ratio",
        "replay_concurrency",
        "isl",
        "osl",
        "cached_prefix_tokens",
        "request_count",
        "turns_per_session",
        "shared_prefix_ratio",
        "num_prefix_groups",
        "inter_turn_delay_ms",
        "request_rate",
        "arrival_interval_ms",
        "arrival_seed",
        "concurrency",
        "num_request_ratio",
        "kv_load_ratio",
        "max_sim_time_ms",
        "agentic_lanes",
        "agentic_snapshot",
        "agentic_warmup",
    }
)

_AIC_TIMING_FIELD_ALIASES = {
    "backend_version": ("backend_version", "aic_backend_version"),
    "pp": ("aic_pp_size",),
    "moe_tp_size": ("moe_tp_size", "aic_moe_tp_size"),
    "moe_ep_size": ("moe_ep_size", "aic_moe_ep_size"),
    "gemm_dtype": ("gemm_dtype", "aic_gemm_dtype"),
    "moe_dtype": ("moe_dtype", "aic_moe_dtype"),
    "fmha_dtype": ("fmha_dtype", "aic_fmha_dtype"),
    "fpm_fmha_dtype": ("fpm_fmha_dtype", "aic_fpm_fmha_dtype"),
    "kv_cache_dtype": ("kv_cache_dtype", "aic_kv_cache_dtype"),
    "comm_dtype": ("comm_dtype", "aic_comm_dtype"),
    "systems_path": ("systems_path",),
    "forward_model": ("forward_model", "aic_forward_model"),
    "fpm_parquet_path": ("fpm_parquet_path", "aic_fpm_parquet_path"),
    "moe_backend": ("aic_moe_backend",),
    "attention_backend": ("aic_attention_backend",),
    "enable_eplb": ("aic_enable_eplb",),
    "wideep_num_slots": ("aic_wideep_num_slots",),
    "decoder_replay": ("decoder_replay", "aic_decoder_replay"),
    "database_mode": ("database_mode", "aic_database_mode"),
    "enable_shared_layer": ("enable_shared_layer", "shared_layer", "aic_enable_shared_layer"),
    "strict_provenance": ("strict_provenance", "aic_strict_provenance"),
    "decode_workload_distribution": ("decode_workload_distribution", "aic_decode_workload_distribution"),
}

_AIC_FORWARD_MODELS = frozenset({"op_level", "fpm"})


class RunnerUnavailableError(RuntimeError):
    """The runtime needed for a requested replay stack is not installed."""


class InvalidRunnerError(RuntimeError):
    """An installed replay runtime does not implement the runner contract."""


@runtime_checkable
class EngineReplayRuntime(Protocol):
    """Serialized execution seam used by the engine-only Runner."""

    def run_replay_json(self, execution_spec_json: str) -> str:
        """Run one serialized execution spec and return serialized report JSON."""


@dataclass(frozen=True)
class AFDCompanionTiming:
    """Static opposite-phase timing used by an `afd+pd` replay."""

    phase: AFDPhase | str
    latency_ms: float
    batch_capacity_per_worker: int
    workers: int
    provenance: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = AFDPhase(self.phase)
        if phase is AFDPhase.BOTH:
            raise ValueError("AFD companion timing must describe prefill or decode")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms <= 0.0
        ):
            raise ValueError("AFD companion latency_ms must be finite and positive")
        _positive_int(self.batch_capacity_per_worker, "AFD companion batch_capacity_per_worker")
        _positive_int(self.workers, "AFD companion workers")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    @property
    def total_batch_capacity(self) -> int:
        return self.batch_capacity_per_worker * self.workers


@runtime_checkable
class AFDCompanionPerformanceModel(Protocol):
    """Measure the regular P/D side of an `afd+pd` deployment."""

    def measure(self, spec: ReplaySpec) -> AFDCompanionTiming: ...


class AICAFDCompanionPerformanceModel:
    """Use AIC static estimation for the regular companion phase."""

    def __init__(self, estimator: Any | None = None) -> None:
        self._estimator = estimator

    def measure(self, spec: ReplaySpec) -> AFDCompanionTiming:
        deployment = spec.backend_deployment
        role = _afd_companion_role(deployment)
        raw_args = deployment.prefill_engine_args if role == "prefill" else deployment.decode_engine_args
        args = _required_engine_args(raw_args, role)
        workload = spec.workload
        isl = _positive_int(workload.get("isl"), "isl")
        osl = _positive_int(workload.get("osl"), "osl")
        token_capacity = _positive_int(args.get("max_num_batched_tokens"), f"{role}_max_num_batched_tokens")
        sequence_capacity = _positive_int(args.get("max_num_seqs"), f"{role}_max_num_seqs")
        tokens_per_sequence = isl if role == "prefill" else 1
        batch_capacity = min(sequence_capacity, max(1, token_capacity // tokens_per_sequence))
        workers = deployment.num_prefill_workers if role == "prefill" else deployment.num_decode_workers
        workers = _positive_int(workers, f"num_{role}_workers")

        timing_model = args.get("timing_model")
        if isinstance(timing_model, Mapping) and timing_model.get("type") == "fixed":
            key = "prefill_ms" if role == "prefill" else "decode_ms"
            latency = _positive_number(timing_model.get(key), f"{role} timing_model.{key}")
            return AFDCompanionTiming(
                phase=role,
                latency_ms=latency,
                batch_capacity_per_worker=batch_capacity,
                workers=workers,
                provenance={"provider": "fixed", "field": key},
            )

        timing_overrides = _pop_aic_timing_overrides(dict(args), role)
        estimator = self._estimator
        if estimator is None:
            from aisimulate.legacy_cli.api import cli_estimate

            estimator = cli_estimate
        prefix = f"{role}_"
        parallel = deployment.parallel_config
        forward_model = timing_overrides.get("forward_model", "op_level")
        kwargs: dict[str, Any] = {
            "mode": "static_ctx" if role == "prefill" else "static_gen",
            "backend_name": deployment.backend,
            "backend_version": deployment.backend_version,
            "forward_model": forward_model,
            "isl": isl,
            "osl": osl,
            "batch_size": batch_capacity,
            "tp_size": _positive_int(parallel.get(f"{prefix}tp"), f"parallel_config.{prefix}tp"),
            "pp_size": _positive_int(parallel.get(f"{prefix}pp", 1), f"parallel_config.{prefix}pp"),
            "attention_dp_size": _positive_int(
                parallel.get(f"{prefix}attention_dp", 1),
                f"parallel_config.{prefix}attention_dp",
            ),
        }
        for source, target in (
            (f"{prefix}moe_tp", "moe_tp_size"),
            (f"{prefix}moe_ep", "moe_ep_size"),
        ):
            if parallel.get(source) is not None:
                kwargs[target] = _positive_int(parallel[source], f"parallel_config.{source}")
        # Timing identity must agree with deployment topology and provenance.
        for identity_field, parameter in (
            ("backend_version", "backend_version"),
            ("pp", "pp_size"),
            ("moe_tp_size", "moe_tp_size"),
            ("moe_ep_size", "moe_ep_size"),
        ):
            if identity_field in timing_overrides and timing_overrides[identity_field] != kwargs.get(parameter):
                raise ValueError(
                    f"{role} AFD companion {identity_field}={timing_overrides[identity_field]!r} "
                    f"conflicts with deployment value {kwargs.get(parameter)!r}; "
                    "set identity fields in BackendDeploymentSpec"
                )
        if args.get("aic_nextn") is not None:
            kwargs["nextn"] = _positive_int(args["aic_nextn"], "aic_nextn")
        model_name = args.get("aic_model_path")
        hardware = args.get("aic_system")
        if not isinstance(model_name, str) or not model_name or not isinstance(hardware, str) or not hardware:
            raise ValueError(f"{role} AFD companion requires aic_model_path and aic_system")
        fpm_parquet_path = timing_overrides.get("fpm_parquet_path")
        has_fpm_selector = "fpm_fmha_dtype" in timing_overrides
        metric = "ttft" if role == "prefill" else "tpot"
        source = "aisimulate.legacy_cli.api.cli_estimate"
        try:
            if fpm_parquet_path is not None or has_fpm_selector:
                # The canonical API is the only path that preserves FPM selectors.
                if forward_model != "fpm":
                    raise ValueError("FPM controls require forward_model='fpm'")
                quantization = {}
                for field, parameter in (
                    ("gemm_dtype", "gemm_quant_mode"),
                    ("moe_dtype", "moe_quant_mode"),
                    ("fmha_dtype", "fmha_quant_mode"),
                    ("fpm_fmha_dtype", "fpm_fmha_quant_mode"),
                    ("kv_cache_dtype", "kvcache_quant_mode"),
                    ("comm_dtype", "comm_quant_mode"),
                ):
                    if field in timing_overrides:
                        quantization[parameter] = timing_overrides[field]
                sharded_moe = kwargs.get("moe_tp_size", 1) * kwargs.get("moe_ep_size", 1) > 1
                estimator_config = (
                    {"fpm_interpolation": {"fpm_parquet_path": fpm_parquet_path}}
                    if fpm_parquet_path is not None
                    else {}
                )
                if "decode_workload_distribution" in timing_overrides:
                    estimator_config["op_level"] = {
                        "decode_workload_distribution": timing_overrides["decode_workload_distribution"]
                    }
                config = ForwardPassPerfModelConfig(
                    model=model_name,
                    system=hardware,
                    backend=deployment.backend,
                    backend_version=deployment.backend_version,
                    worker_type=role,
                    tp=kwargs["tp_size"],
                    pp=kwargs["pp_size"],
                    attention_dp=kwargs["attention_dp_size"],
                    moe_tp_size=kwargs.get("moe_tp_size") if sharded_moe else None,
                    moe_ep_size=kwargs.get("moe_ep_size") if sharded_moe else None,
                    nextn=kwargs.get("nextn", 0),
                    kv_block_size=args.get("block_size"),
                    estimation_mode="fpm_interpolation",
                    estimator_config=estimator_config,
                    systems_paths=(timing_overrides["systems_path"],) if "systems_path" in timing_overrides else (),
                    **quantization,
                    **{
                        name: timing_overrides[name]
                        for name in (
                            "attention_backend",
                            "moe_backend",
                            "enable_eplb",
                            "wideep_num_slots",
                            "decoder_replay",
                            "database_mode",
                            "enable_shared_layer",
                            "strict_provenance",
                        )
                        if name in timing_overrides
                    },
                )
                model = RustForwardPassPerfModel.best_available(config)
                try:
                    latency = model.static_phase_latency(
                        batch_size=batch_capacity, input_tokens=isl, output_tokens=osl, prefill=role == "prefill"
                    )
                    if role == "decode":
                        latency /= max(1, osl - 1)
                finally:
                    model.close()
                raw = {metric: latency}
                source = "aisimulate_core.sdk.rust_engine_step.RustForwardPassPerfModel"
            else:
                result = estimator(model_name, hardware, **kwargs)
                raw = getattr(result, "raw", None)
        except Exception as exc:
            raise InvalidRunnerError(
                f"AIC could not measure the AFD {role} companion: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise InvalidRunnerError("AIC AFD companion estimate did not return a result mapping")
        latency = _positive_number(raw.get(metric), f"AIC AFD companion {metric}")
        return AFDCompanionTiming(
            phase=role,
            latency_ms=latency,
            batch_capacity_per_worker=batch_capacity,
            workers=workers,
            provenance={
                "provider": "aic",
                "source": source,
                "backend_version": deployment.backend_version,
                "forward_model": forward_model,
                **({"fpm_parquet_path": fpm_parquet_path} if fpm_parquet_path is not None else {}),
                **({"fpm_fmha_dtype": timing_overrides["fpm_fmha_dtype"]} if has_fpm_selector else {}),
                "metric": metric,
            },
        )


@dataclass(frozen=True)
class EngineReplayRunnerFactory:
    """Create reusable engine-only Runners for Sweeper candidates.

    The optional runtime is a test seam. Production factories leave it unset so
    the dataclass remains process-serializable and each worker loads its local
    compiled runtime lazily.
    """

    trace_block_size: int = 512
    runtime: EngineReplayRuntime | None = field(default=None, repr=False, compare=False)
    afd_companion_model: AFDCompanionPerformanceModel | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            replay_spec_api_version=1,
            supported_backend_topologies=_SUPPORTED_BACKEND_TOPOLOGIES,
            supports_disaggregated_attention_dp=True,
            supports_analytical_epd=True,
            supports_cached_prefix_tokens=True,
            supports_mtp_expected_acceptance=True,
            supported_engine_model_controls=ENGINE_MODEL_CONTROL_FIELDS,
            supports_state_cache=True,
            supported_trace_formats=(
                "mooncake",
                "mooncake-delta",
                "agentic_mooncake",
                "applied_compute_agentic",
                "dynamo",
                "weka",
            ),
            supports_agentic_lanes=True,
            supports_agentic_snapshots=True,
            supports_agentic_warmup=True,
            supported_agentic_topologies=("agg", "disagg"),
            supported_agentic_backends=("vllm", "sglang"),
            supports_agentic_host_offload=False,
            supports_agentic_speculative_decoding=False,
            agentic_qualification="functional_only",
        )

    def create(self, worker_id: int) -> EngineReplayRunner:
        return EngineReplayRunner(
            worker_id=worker_id,
            capabilities=self.capabilities(),
            trace_block_size=self.trace_block_size,
            runtime=self.runtime,
            afd_companion_model=self.afd_companion_model,
        )


@dataclass
class EngineReplayRunner:
    """Execute one candidate with the AISimulate Engine/Replayer composition."""

    worker_id: int
    capabilities: RunnerCapabilities
    trace_block_size: int = 512
    runtime: EngineReplayRuntime | None = None
    afd_companion_model: AFDCompanionPerformanceModel | None = None

    def _resolve_runtime(self) -> EngineReplayRuntime:
        if self.runtime is not None:
            return self.runtime

        try:
            runtime = importlib.import_module("aisimulate._runtime")
        except ModuleNotFoundError as exc:
            if exc.name != "aisimulate._runtime":
                raise
            raise RunnerUnavailableError(
                "AISimulate engine replay runtime is not installed. Install a "
                "binary aisimulate package or build its native extension."
            ) from exc
        except ImportError as exc:
            raise RunnerUnavailableError(
                "AISimulate engine replay runtime could not be loaded. Reinstall "
                "the binary aisimulate package or rebuild its native extension."
            ) from exc

        if not isinstance(runtime, EngineReplayRuntime):
            raise InvalidRunnerError("aisimulate._runtime must export run_replay_json(execution_spec_json: str) -> str")
        self.runtime = runtime
        return runtime

    def run(
        self,
        spec: ReplaySpec,
        *,
        output_requirements: ReplayOutputRequirements | None = None,
    ) -> ReplayReport:
        output_requirements = output_requirements or ReplayOutputRequirements()
        if output_requirements.capture_telemetry:
            raise InvalidRunnerError("EngineReplayRunner's JSON runtime does not yet expose replay telemetry")
        if spec.workload.get("source_type") is not None and "length_sampler" in spec.workload:
            raise ValueError("length_sampler requires materialized direct synthetic replay without source_type")
        self.capabilities.require_compatible(spec)
        encoder = spec.backend_deployment.encoder
        if encoder is None and spec.workload.get("images") is not None:
            raise InvalidRunnerError("image workloads require an encoder pool")
        if spec.backend_deployment.deployment_mode in {"afd", "afd+pd"}:
            return _run_afd_replay(
                spec,
                trace_block_size=self.trace_block_size,
                companion_model=self.afd_companion_model,
                include_report=output_requirements.include_raw_report,
                capture_per_request=output_requirements.capture_per_request,
            )
        if encoder is not None:
            from .sweeper.config import OptimizationGoal, Workload
            from .sweeper.epd import apply_encoder_overlay

            if output_requirements.capture_per_request or output_requirements.include_raw_report:
                raise InvalidRunnerError("analytical EPD cannot produce per-request or raw replay reports")
            if spec.adapters or spec.execution_mode != "offline":
                raise InvalidRunnerError("analytical EPD requires offline static pools without adapters")
            goal = OptimizationGoal.model_validate(spec.goal)
            if (
                goal.min_goodput_rps is not None
                or (goal.sla is not None and not goal.requires_aggregate_sla)
                or any(
                    target.value.startswith("goodput")
                    for target in (goal.resolved_pareto_objectives if goal.is_pareto else [goal.target])
                )
            ):
                raise InvalidRunnerError("analytical EPD cannot report per-request goodput")
            workload = Workload.model_validate(spec.workload)
            workload.require_fixed_epd()
            images = workload.images
            if (images.height, images.width, images.count) != (
                encoder.image_height,
                encoder.image_width,
                encoder.image_count,
            ):
                raise InvalidRunnerError("encoder estimate does not match the image workload")
            if encoder.backend != spec.backend_deployment.backend:
                raise InvalidRunnerError("encoder and language backend must match")
            deployment = spec.backend_deployment
            role_args = (
                [deployment.agg_engine_args]
                if deployment.deployment_mode == "agg"
                else [deployment.prefill_engine_args, deployment.decode_engine_args]
            )
            for args in role_args:
                if not args or args.get("aic_model_path") != encoder.model:
                    raise InvalidRunnerError("encoder and language model must match")
                scopes = [args]
                if args.get("rank") is not None:
                    if not isinstance(args["rank"], Mapping):
                        raise InvalidRunnerError("language rank config must be a mapping")
                    scopes.append(args["rank"])
                for scope in scopes:
                    if scope.get("timing_model") is not None or any(
                        scope.get(alias, "op_level") != "op_level"
                        for alias in _AIC_TIMING_FIELD_ALIASES["forward_model"]
                    ):
                        raise InvalidRunnerError("analytical EPD requires op_level language timing")
                    if scope.get("startup_time") not in (None, 0.0):
                        raise InvalidRunnerError("analytical EPD requires static worker pools")
            original_spec = spec
            spec = replace(spec, workload={**spec.workload, "isl": spec.workload["isl"] + encoder.visual_tokens})
        memory_diagnostics = {} if output_requirements.capture_memory_diagnostics else None
        execution_spec = _materialize_engine_execution_spec(
            spec,
            trace_block_size=self.trace_block_size,
            record_per_request=output_requirements.capture_per_request,
            memory_diagnostics=memory_diagnostics,
        )
        if output_requirements.capture_performance_diagnostics:
            if "spec" not in execution_spec:
                execution_spec = {"spec": execution_spec}
            execution_spec["capture_performance_diagnostics"] = True
        execution_spec_json = json.dumps(
            execution_spec,
            allow_nan=False,
            separators=(",", ":"),
        )
        try:
            report_json = self._resolve_runtime().run_replay_json(execution_spec_json)
        except MemoryError as error:
            # Resource-aware installations classify a report storage failure as host
            # exhaustion, not a failed candidate. Older SDKs retain MemoryError.
            try:
                from .resources import ResourceLimitError
            except ImportError:
                raise error from None
            raise ResourceLimitError(str(error)) from error
        if not isinstance(report_json, str):
            raise InvalidRunnerError("AISimulate engine replay runtime report must be a JSON string")
        try:
            report = json.loads(report_json)
        except json.JSONDecodeError as exc:
            raise InvalidRunnerError("AISimulate engine replay runtime returned invalid report JSON") from exc
        if not isinstance(report, Mapping):
            raise InvalidRunnerError("AISimulate engine replay runtime report must be a JSON object")
        resolved_basis = report.get("weka_nested_timestamp_basis")
        if isinstance(resolved_basis, str):
            logger.info(
                "The complete Weka corpus uses resolved nested timestamp basis %r; auto selection, "
                "when requested, is a corpus-wide heuristic and the resolved basis is included in source identity",
                resolved_basis,
            )
        if memory_diagnostics is not None:
            report = {**report, "memory_diagnostics": memory_diagnostics}
        normalized = _normalize_engine_replay_report(
            report,
            include_native_report=(
                output_requirements.include_raw_report
                or output_requirements.capture_per_request
                or output_requirements.capture_memory_diagnostics
                or output_requirements.capture_performance_diagnostics
            ),
        )
        if encoder is not None:
            normalized = apply_encoder_overlay(normalized, original_spec)
            if memory_diagnostics is not None:
                memory_diagnostics["encoder"] = {
                    "scope": "capacity_estimate_per_rank",
                    "stage": "before_native_capacity_adjustments",
                    "status": "unavailable",
                    "unavailable_reason": "analytical EPD does not export an encoder memory component estimate",
                }
                # Capacity estimates remain valid across the overlay. Raw language
                # timing/records do not describe the combined EPD workload.
                normalized = ReplayReport(
                    metrics=normalized.metrics,
                    metadata={**normalized.metadata, "memory_diagnostics": memory_diagnostics},
                )
        return normalized

    def close(self) -> None:
        """Release worker-local resources.

        The current extension owns no persistent resources beyond its imported
        module, so closing is intentionally a no-op.
        """


def _afd_companion_role(deployment) -> str:
    has_prefill = deployment.prefill_engine_args is not None
    has_decode = deployment.decode_engine_args is not None
    if has_prefill == has_decode:
        raise ValueError("afd+pd requires exactly one prefill or decode companion engine")
    return "prefill" if has_prefill else "decode"


def _afd_topology(deployment) -> AFDTopology:
    raw = deployment.parallel_config.get("afd")
    if not isinstance(raw, Mapping):
        raise ValueError("AFD deployment requires parallel_config.afd")
    values = dict(raw)
    values.pop("ffn_tp", None)
    topology = AFDTopology(**values)
    if topology.adapter_topology != deployment.deployment_mode:
        raise ValueError(f"AFD topology advertises {topology.adapter_topology!r}, not {deployment.deployment_mode!r}")
    return topology


def _afd_layer_measurements(deployment) -> tuple[AFDLayerTimes, ...]:
    metadata = deployment.performance_model_metadata.get("afd")
    if not isinstance(metadata, Mapping):
        raise ValueError("AFD deployment is missing performance_model_metadata.afd")
    if metadata.get("measurement_required") is not False:
        raise ValueError("AFD deployment performance measurement is unresolved")
    raw_measurements = metadata.get("measurements")
    if not isinstance(raw_measurements, list) or not raw_measurements:
        raise ValueError("AFD deployment requires a non-empty measurements list")
    measurements: list[AFDLayerTimes] = []
    for index, raw in enumerate(raw_measurements):
        if not isinstance(raw, Mapping):
            raise TypeError(f"AFD measurement {index} must be a mapping")
        try:
            measurements.append(
                AFDLayerTimes(
                    phase=raw["phase"],
                    attention_ms=raw["attention_ms"],
                    ffn_ms=raw["ffn_ms"],
                    a_to_f_ms=raw["a_to_f_ms"],
                    f_to_a_ms=raw["f_to_a_ms"],
                    num_layers=raw["num_layers"],
                    provenance=raw.get("provenance", {}),
                )
            )
        except KeyError as exc:
            raise ValueError(f"AFD measurement {index} is missing {exc.args[0]!r}") from exc
    return tuple(measurements)


def _afd_total_gpus(deployment, topology: AFDTopology) -> int:
    expected = topology.total_gpus
    if deployment.deployment_mode == "afd+pd":
        role = _afd_companion_role(deployment)
        prefix = f"{role}_"
        parallel = deployment.parallel_config
        replicas = _positive_int(
            parallel.get(f"{prefix}replicas"),
            f"parallel_config.{prefix}replicas",
        )
        tp = _positive_int(parallel.get(f"{prefix}tp"), f"parallel_config.{prefix}tp")
        dp = _positive_int(
            parallel.get(f"{prefix}attention_dp", 1),
            f"parallel_config.{prefix}attention_dp",
        )
        pp = _positive_int(
            parallel.get(f"{prefix}pp", 1),
            f"parallel_config.{prefix}pp",
        )
        expected += replicas * tp * dp * pp
    provenance = deployment.parallel_config.get("afd_provenance")
    if isinstance(provenance, Mapping):
        accounting = provenance.get("gpu_accounting")
        if isinstance(accounting, Mapping) and accounting.get("total_gpus") is not None:
            recorded = _positive_int(accounting["total_gpus"], "AFD total_gpus")
            if recorded != expected:
                raise ValueError(f"AFD provenance total_gpus={recorded} conflicts with topology accounting={expected}")
    return expected


def _run_afd_phase(
    engine: AFDForegroundEngine,
    *,
    phase: AFDPhase,
    start_ms: float,
    input_length: int,
    output_length: int,
    passes: int,
) -> tuple[float, float, int]:
    cursor = start_ms
    pass_latency_ms = 0.0
    interval_count = 0
    for _ in range(passes):
        planned = engine.execute_pass(
            phase=phase,
            now_ms=cursor,
            input_length=input_length,
            output_length=output_length,
        )
        completed = engine.complete_pass(planned.pass_id, now_ms=planned.end_ms)
        cursor = completed.completed_at_ms
        pass_latency_ms = completed.pass_latency_ms
        interval_count += len(planned.intervals)
    return cursor, pass_latency_ms, interval_count


def _request_passes_sla(record: Mapping[str, JSONValue], sla: Mapping[str, JSONValue]) -> bool:
    comparisons = (
        ("ttft_ms", "ttft_ms"),
        ("itl_ms", "tpot_ms"),
        ("e2e_ms", "e2e_latency_ms"),
    )
    return all(sla.get(bound) is None or float(record[metric]) <= float(sla[bound]) for bound, metric in comparisons)


def _run_afd_replay(
    spec: ReplaySpec,
    *,
    trace_block_size: int,
    companion_model: AFDCompanionPerformanceModel | None,
    include_report: bool,
    capture_per_request: bool,
) -> ReplayReport:
    deployment = spec.backend_deployment
    topology = _afd_topology(deployment)
    measurements = _afd_layer_measurements(deployment)
    if spec.workload.get("trace_path") is not None:
        raise ValueError(
            "AFD engine replay requires concrete synthetic isl/osl measurements; trace replay is not supported"
        )
    if spec.workload.get("random_range_ratio", 1.0) != 1.0:
        raise ValueError("AFD engine replay requires random_range_ratio=1.0 to match measured sequence lengths")
    if spec.workload.get("max_sim_time_ms") is not None:
        raise ValueError("AFD engine replay does not yet support max_sim_time_ms")
    requests, max_in_flight = _materialize_requests(spec, trace_block_size)
    if not requests:
        raise ValueError("AFD engine replay requires at least one request")
    input_length = _positive_int(spec.workload.get("isl"), "isl")
    output_length = _positive_int(spec.workload.get("osl"), "osl")
    engine = AFDForegroundEngine(topology, measurements)

    companion: AFDCompanionTiming | None = None
    if deployment.deployment_mode == "afd":
        if topology.phase is not AFDPhase.BOTH:
            raise ValueError("pure AFD replay requires phase='both'; use afd+pd for a single AFD phase")
        batch_capacity = topology.total_batch_size
    else:
        if topology.phase is AFDPhase.BOTH:
            raise ValueError("afd+pd replay requires one concrete AFD phase")
        companion = (companion_model or AICAFDCompanionPerformanceModel()).measure(spec)
        expected_companion = AFDPhase.DECODE if topology.phase is AFDPhase.PREFILL else AFDPhase.PREFILL
        if companion.phase is not expected_companion:
            raise ValueError(
                f"AFD companion phase {companion.phase.value!r} does not complement {topology.phase.value!r}"
            )
        batch_capacity = min(topology.total_batch_size, companion.total_batch_capacity)
    if max_in_flight is not None:
        batch_capacity = min(batch_capacity, max_in_flight)
    batch_capacity = _positive_int(batch_capacity, "AFD replay batch capacity")

    ordered_requests = sorted(requests, key=lambda request: float(request["arrival_time_ms"]))
    # Each slot admits one request and becomes available again only when that
    # request finishes both phases, including time queued between the pools.
    admission_slots = [0.0] * min(max_in_flight, len(requests)) if max_in_flight is not None else None
    request_records: list[dict[str, JSONValue]] = []
    next_request = 0
    afd_available_ms = 0.0
    companion_available_ms = 0.0
    if companion is not None:
        raw_args = (
            deployment.prefill_engine_args if companion.phase is AFDPhase.PREFILL else deployment.decode_engine_args
        )
        companion_available_ms = _startup_delay_ms(_required_engine_args(raw_args, companion.phase.value))
    afd_passes = 0
    afd_intervals = 0
    batch_count = 0
    while next_request < len(ordered_requests):
        first_phase_available = (
            afd_available_ms
            if deployment.deployment_mode == "afd" or topology.phase is AFDPhase.PREFILL
            else companion_available_ms
        )
        batch_start = max(
            first_phase_available,
            _nonnegative_time(ordered_requests[next_request]["arrival_time_ms"], "arrival_time_ms"),
            admission_slots[0] if admission_slots is not None else 0.0,
        )
        batch = []
        while next_request < len(ordered_requests) and len(batch) < batch_capacity:
            request = ordered_requests[next_request]
            arrival = _nonnegative_time(request["arrival_time_ms"], "arrival_time_ms")
            if admission_slots is not None:
                if not admission_slots:
                    break
                arrival = max(arrival, admission_slots[0])
            if arrival > batch_start and batch:
                break
            if arrival > batch_start:
                batch_start = arrival
            if (
                _positive_int(request["input_tokens"], "request input_tokens") != input_length
                or _positive_int(request["output_tokens"], "request output_tokens") != output_length
            ):
                raise ValueError("AFD request lengths must match the measured workload isl/osl")
            if admission_slots is not None:
                heapq.heappop(admission_slots)
                request["arrival_time_ms"] = arrival
            batch.append(request)
            next_request += 1
        batch_count += 1

        decode_passes = max(output_length - 1, 0)
        if deployment.deployment_mode == "afd":
            prefill_start = max(batch_start, afd_available_ms)
            prefill_end, _, interval_count = _run_afd_phase(
                engine,
                phase=AFDPhase.PREFILL,
                start_ms=prefill_start,
                input_length=input_length,
                output_length=output_length,
                passes=1,
            )
            decode_end, _, decode_intervals = _run_afd_phase(
                engine,
                phase=AFDPhase.DECODE,
                start_ms=prefill_end,
                input_length=input_length,
                output_length=output_length,
                passes=decode_passes,
            )
            afd_available_ms = decode_end
            afd_passes += 1 + decode_passes
            afd_intervals += interval_count + decode_intervals
        elif topology.phase is AFDPhase.PREFILL:
            assert companion is not None
            prefill_start = max(batch_start, afd_available_ms)
            prefill_end, _, interval_count = _run_afd_phase(
                engine,
                phase=AFDPhase.PREFILL,
                start_ms=prefill_start,
                input_length=input_length,
                output_length=output_length,
                passes=1,
            )
            afd_available_ms = prefill_end
            decode_end = prefill_end
            if decode_passes:
                decode_start = max(prefill_end, companion_available_ms)
                decode_end = decode_start + companion.latency_ms * decode_passes
                companion_available_ms = decode_end
            afd_passes += 1
            afd_intervals += interval_count
        else:
            assert companion is not None
            prefill_start = max(batch_start, companion_available_ms)
            prefill_end = prefill_start + companion.latency_ms
            companion_available_ms = prefill_end
            decode_end = prefill_end
            if decode_passes:
                decode_start = max(prefill_end, afd_available_ms)
                decode_end, _, interval_count = _run_afd_phase(
                    engine,
                    phase=AFDPhase.DECODE,
                    start_ms=decode_start,
                    input_length=input_length,
                    output_length=output_length,
                    passes=decode_passes,
                )
                afd_available_ms = decode_end
                afd_passes += decode_passes
                afd_intervals += interval_count

        for request in batch:
            if admission_slots is not None:
                heapq.heappush(admission_slots, decode_end)
            arrival = float(request["arrival_time_ms"])
            request_records.append(
                {
                    "id": str(request["id"]),
                    "arrival_time_ms": arrival,
                    "ttft_ms": prefill_end - arrival,
                    "tpot_ms": (decode_end - prefill_end) / decode_passes if decode_passes else 0.0,
                    "e2e_latency_ms": decode_end - arrival,
                    "output_tokens": output_length,
                }
            )

    first_arrival = min(float(request["arrival_time_ms"]) for request in ordered_requests)
    end_ms = max(float(record["arrival_time_ms"]) + float(record["e2e_latency_ms"]) for record in request_records)
    duration_ms = end_ms - first_arrival
    if duration_ms <= 0.0:
        raise InvalidRunnerError("AFD replay produced a non-positive duration")
    completed = len(request_records)
    output_tokens = completed * output_length
    sla = _materialize_sla(spec)
    good_output_tokens = sum(
        int(record["output_tokens"]) for record in request_records if _request_passes_sla(record, sla)
    )
    total_gpus = _afd_total_gpus(deployment, topology)
    mean_ttft = sum(float(record["ttft_ms"]) for record in request_records) / completed
    mean_tpot = sum(float(record["tpot_ms"]) for record in request_records) / completed
    mean_e2e = sum(float(record["e2e_latency_ms"]) for record in request_records) / completed
    input_tokens = completed * input_length
    duration_s = duration_ms / 1_000.0
    metrics = {
        "duration_ms": duration_ms,
        "num_requests": float(completed),
        "total_input_tokens": float(input_tokens),
        "total_output_tokens": float(output_tokens),
        "request_throughput_rps": completed / duration_s,
        "input_throughput_tok_s": input_tokens / duration_s,
        "output_throughput_tok_s": output_tokens * 1_000.0 / duration_ms,
        "total_throughput_tok_s": (input_tokens + output_tokens) / duration_s,
        "gpu_hours": total_gpus * duration_ms / 3_600_000.0,
        "num_ttft_samples": float(completed),
        "num_tpot_samples": float(completed if output_length > 1 else 0),
        "num_e2e_latency_samples": float(completed),
        "mean_ttft_ms": mean_ttft,
        "mean_tpot_ms": mean_tpot,
        "mean_e2e_latency_ms": mean_e2e,
        "mean_output_token_throughput_per_user": 1_000.0 / mean_tpot if mean_tpot > 0.0 else 0.0,
        "completed_requests": float(completed),
        "error_rate": 0.0,
        "num_total_gpus": float(total_gpus),
    }
    if sla:
        metrics["goodput_completed_requests"] = float(
            sum(1 for record in request_records if _request_passes_sla(record, sla))
        )
        metrics["goodput_request_throughput_rps"] = metrics["goodput_completed_requests"] / duration_s
        metrics["goodput_output_throughput_tok_s"] = good_output_tokens / duration_s
    metrics.update(normalize_power_summary({}))
    summary: dict[str, JSONValue] = {
        "executor": "afd_foreground",
        "deployment_mode": deployment.deployment_mode,
        "phase": topology.phase.value,
        "batches": batch_count,
        "afd_passes": afd_passes,
        "afd_stage_intervals": afd_intervals,
        "batch_capacity": batch_capacity,
        "companion": companion.provenance if companion is not None else None,
    }
    metadata: dict[str, JSONValue] = {"afd_replay": summary}
    native_report: dict[str, JSONValue] = {
        "summary": dict(metrics),
        "afd_replay": summary,
    }
    if capture_per_request:
        # Preserve the runner-level compatibility projection while also exposing
        # the public CLI's canonical native-report shape.
        metadata["per_request"] = request_records
        native_report["per_request"] = request_records
    if include_report or capture_per_request:
        metadata["native_report"] = native_report
    if include_report:
        metadata["afd_report"] = {"metrics": metrics, **summary}
    metrics.update(normalize_power_summary(metrics))
    metadata["power"] = power_metadata(metrics)
    return ReplayReport(metrics=metrics, metadata=metadata)


def _materialize_engine_execution_spec(
    spec: ReplaySpec,
    *,
    trace_block_size: int,
    record_per_request: bool,
    memory_diagnostics: dict[str, Any] | None = None,
) -> dict[str, JSONValue]:
    """Translate the public Runner input into the Rust replay wire schema.

    This is the only high-level-to-execution materializer. The compiled runtime
    deliberately knows nothing about CLI, Sweeper, or provider discovery.
    """

    deployment = spec.backend_deployment
    deployment_mode = deployment.deployment_mode
    execution_model: str | None = None
    if deployment_mode == "agg":
        raw_engine_args = _required_engine_args(deployment.agg_engine_args, "aggregated")
        execution_model = _execution_target_model(deployment, "aggregated", raw_engine_args)
        engine = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_engine_args,
            "aggregated",
            memory_diagnostics=memory_diagnostics,
        )
        _require_parallel_match(
            deployment.parallel_config,
            "replicas",
            deployment.num_workers,
            "num_workers",
        )
        topology: dict[str, JSONValue] = {
            "kind": "aggregated",
            "workers": {
                "initial_workers": _positive_int(deployment.num_workers, "num_workers"),
                "startup_delay_ms": _startup_delay_ms(raw_engine_args),
            },
        }
    elif deployment_mode == "disagg":
        raw_prefill = _required_engine_args(deployment.prefill_engine_args, "prefill")
        raw_decode = _required_engine_args(deployment.decode_engine_args, "decode")
        prefill = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_prefill,
            "prefill",
            memory_diagnostics=memory_diagnostics,
        )
        decode = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_decode,
            "decode",
            memory_diagnostics=memory_diagnostics,
        )
        _require_parallel_match(
            deployment.parallel_config,
            "prefill_replicas",
            deployment.num_prefill_workers,
            "num_prefill_workers",
        )
        _require_parallel_match(
            deployment.parallel_config,
            "decode_replicas",
            deployment.num_decode_workers,
            "num_decode_workers",
        )
        prefill_backend = prefill["rank"].get("backend", "vllm")
        decode_backend = decode["rank"].get("backend", "vllm")
        if prefill_backend != decode_backend:
            raise ValueError(
                f"disaggregated prefill and decode must use the same backend: {prefill_backend!r} != {decode_backend!r}"
            )
        engine = {"prefill": prefill, "decode": decode}
        topology = {
            "kind": "disaggregated",
            "prefill": {
                "initial_workers": _positive_int(deployment.num_prefill_workers, "num_prefill_workers"),
                "startup_delay_ms": _startup_delay_ms(raw_prefill),
            },
            "decode": {
                "initial_workers": _positive_int(deployment.num_decode_workers, "num_decode_workers"),
                "startup_delay_ms": _startup_delay_ms(raw_decode),
            },
            "handoff_latency_ms": 0.0,
        }
    else:
        raise ValueError(f"engine replay deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}")

    use_workload_driver = spec.workload.get("source_type") is not None
    if use_workload_driver:
        requests: list[dict[str, JSONValue]] = []
        max_in_flight = _configured_in_flight_cap(spec)
    else:
        requests, max_in_flight = _materialize_requests(spec, trace_block_size)
    sla = _materialize_sla(spec)
    max_sim_time_ms = spec.workload.get("max_sim_time_ms")
    if max_sim_time_ms is not None:
        max_sim_time_ms = _nonnegative_time(max_sim_time_ms, "max_sim_time_ms")

    execution_spec: dict[str, JSONValue] = {
        "version": 1,
        "topology": topology,
        "engine": engine,
        "adapters": {
            "placement": {
                "provider": "round_robin",
                "config": None,
            },
            "scaling": {
                "provider": "none",
                "config": None,
            },
        },
        "max_sim_time_ms": max_sim_time_ms,
        "max_in_flight": max_in_flight,
        "record_per_request": record_per_request,
        "sla": sla,
        "requests": requests,
    }
    if use_workload_driver:
        traffic = {
            key: value for key, value in spec.workload.items() if key in _RUNTIME_TRAFFIC_FIELDS and value is not None
        }
        if traffic.get("trace_format") not in {"dynamo", "weka"}:
            traffic.setdefault("trace_block_size", trace_block_size)
        trace_format = traffic.get("trace_format")
        if trace_format in {"agentic_mooncake", "dynamo", "weka"}:
            requires_agentic_model = trace_format != "dynamo" or traffic.get("agentic_lanes") is not None
            if deployment_mode == "disagg":
                prefill_model = _execution_target_model(deployment, "prefill", raw_prefill)
                decode_model = _execution_target_model(deployment, "decode", raw_decode)
                if requires_agentic_model and prefill_model != decode_model:
                    raise ValueError("agentic prefill and decode must use the same configured target model")
                if prefill_model == decode_model:
                    execution_model = prefill_model
                if requires_agentic_model:
                    for role, descriptor in (("prefill", prefill), ("decode", decode)):
                        timing = descriptor["rank"].get("timing_model")
                        if (
                            isinstance(timing, Mapping)
                            and timing.get("type") == "external"
                            and timing.get("provider") == "aic"
                            and isinstance(timing.get("config"), Mapping)
                            and timing["config"].get("model") != execution_model
                        ):
                            raise ValueError(f"agentic {role} AIC timing model must match the configured target model")
            if requires_agentic_model and execution_model is None:
                raise ValueError("agentic execution requires a configured target model")
            # Dynamo may contain standard or agentic requests; native validates the loaded kind.
            if execution_model is not None:
                traffic["execution_model"] = execution_model
        return {"spec": execution_spec, "traffic": traffic}
    return execution_spec


def _execution_target_model(
    deployment: BackendDeploymentSpec,
    role: str,
    raw_engine_args: Mapping[str, JSONValue],
) -> str | None:
    """Resolve the deployment model identity independently of timing implementation."""

    metadata = deployment.performance_model_metadata.get(role)
    if isinstance(metadata, Mapping):
        config = metadata.get("config")
        if isinstance(config, Mapping):
            model = config.get("model", config.get("model_path"))
            if isinstance(model, str) and model.strip():
                return model.strip()
    timing = raw_engine_args.get("timing_model")
    if isinstance(timing, Mapping) and timing.get("provider") == "aic":
        config = timing.get("config")
        if isinstance(config, Mapping):
            model = config.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
    model = raw_engine_args.get("aic_model_path")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _configured_in_flight_cap(spec: ReplaySpec) -> int | None:
    workload = spec.workload
    if workload.get("source_type") == "trace":
        value = workload.get("replay_concurrency")
    else:
        value = spec.concurrency
        if value is None:
            value = workload.get("concurrency")
    return _positive_int(value, "concurrency") if value is not None else None


def _pop_aic_timing_overrides(rank: dict[str, JSONValue], role: str) -> dict[str, JSONValue]:
    aic_timing_overrides: dict[str, JSONValue] = {}
    for target, aliases in _AIC_TIMING_FIELD_ALIASES.items():
        configured = [alias for alias in aliases if alias in rank]
        if len(configured) > 1:
            names = ", ".join(configured)
            raise ValueError(f"engine provider {role} config duplicates AIC field {target}: {names}")
        if not configured:
            continue
        value = rank.pop(configured[0])
        if target == "decode_workload_distribution" and value is None:
            continue
        if target in {"pp", "moe_tp_size", "moe_ep_size", "wideep_num_slots"}:
            value = _positive_int(value, f"engine provider {role} {target}")
        elif target in {"enable_eplb", "decoder_replay", "enable_shared_layer", "strict_provenance"}:
            if not isinstance(value, bool):
                raise ValueError(f"engine provider {role} {target} must be a boolean")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"engine provider {role} {target} must be a string")
        if target == "forward_model" and value not in _AIC_FORWARD_MODELS:
            raise ValueError(
                f"engine provider {role} forward_model must be one of {sorted(_AIC_FORWARD_MODELS)}, got {value!r}"
            )
        aic_timing_overrides[target] = value

    return aic_timing_overrides


def _required_engine_args(payload: dict[str, JSONValue] | None, role: str) -> dict[str, JSONValue]:
    if payload is None:
        raise ValueError(f"ReplaySpec is missing {role} engine arguments")
    return dict(payload)


def _startup_delay_ms(payload: Mapping[str, JSONValue]) -> float:
    value = payload.get("startup_time", 0.0)
    return 1_000.0 * _nonnegative_time(value, "startup_time")


def _materialize_requests(spec: ReplaySpec, trace_block_size: int) -> tuple[list[dict[str, JSONValue]], int | None]:
    workload = spec.workload
    trace_path = workload.get("trace_path")
    if trace_path is not None:
        if not isinstance(trace_path, str) or not trace_path:
            raise TypeError("trace_path must be a non-empty string")
        if workload.get("random_range_ratio", 1.0) != 1.0 or workload.get("random_seed", 0) != 0:
            raise ValueError("random_range_ratio and random_seed only apply to synthetic replay")
        if workload.get("length_sampler", "python_random") != "python_random":
            raise ValueError("length_sampler only applies to synthetic replay")
        configured_trace_block_size = workload.get("trace_block_size")
        requests = materialize_configured_traffic(
            {
                "trace_path": trace_path,
                "format": workload.get("trace_format", "mooncake"),
                "trace_block_size": _positive_int(
                    trace_block_size if configured_trace_block_size is None else configured_trace_block_size,
                    "trace_block_size",
                ),
                "speedup": _positive_number(
                    workload.get("arrival_speedup_ratio", 1.0),
                    "arrival_speedup_ratio",
                ),
            }
        )
        cap = workload.get("replay_concurrency")
        return requests, (_positive_int(cap, "replay_concurrency") if cap is not None else None)

    isl = _positive_int(workload.get("isl"), "isl")
    osl = _positive_int(workload.get("osl"), "osl")
    _require_supported_synthetic_workload(workload)
    concurrency = spec.concurrency
    if concurrency is None and workload.get("concurrency") is not None:
        concurrency = _positive_int(workload["concurrency"], "concurrency")
    request_rate = workload.get("request_rate")
    rate = _positive_number(request_rate, "request_rate") if request_rate is not None else None
    raw_interval = workload.get("arrival_interval_ms")
    arrival_interval_ms = _nonnegative_time(raw_interval, "arrival_interval_ms") if raw_interval is not None else None
    if rate is not None and arrival_interval_ms is not None:
        raise ValueError("synthetic replay cannot combine request_rate and arrival_interval_ms")
    if concurrency is not None:
        load = float(concurrency)
    elif rate is not None:
        load = rate
    elif arrival_interval_ms is None:
        raise ValueError("synthetic replay requires concurrency, request_rate, or arrival_interval_ms")
    raw_request_count = workload.get("request_count")
    if raw_request_count is not None:
        request_count = _positive_int(raw_request_count, "request_count")
    else:
        if arrival_interval_ms is not None:
            raise ValueError("synthetic replay with arrival_interval_ms requires request_count")
        ratio = _positive_number(workload.get("num_request_ratio"), "num_request_ratio")
        request_count = max(1, round(ratio * load))

    if rate is not None:
        arrival_seed = workload.get("arrival_seed", 42)
        if not isinstance(arrival_seed, int) or isinstance(arrival_seed, bool) or arrival_seed < 0:
            raise ValueError("arrival_seed must be a non-negative integer")
        rng = random.Random(arrival_seed)
        arrival_times = [0.0]
        for _ in range(1, request_count):
            arrival_times.append(arrival_times[-1] + rng.expovariate(rate) * 1_000.0)
    else:
        interval = arrival_interval_ms or 0.0
        arrival_times = [index * interval for index in range(request_count)]

    random_range_ratio = _random_range_ratio(workload.get("random_range_ratio", 1.0))
    random_seed = _random_seed(workload.get("random_seed", 0))
    sampler = workload.get("length_sampler", "python_random")
    if sampler == "numpy_random_state":
        if random_seed > 0xFFFF_FFFF:
            raise ValueError("numpy_random_state random_seed must be an unsigned 32-bit integer")
        length_rng = np.random.RandomState(random_seed)
    elif sampler == "python_random":
        length_rng = random.Random(random_seed)
    else:
        raise ValueError(f"unsupported length_sampler: {sampler!r}")
    # Follow InferenceX's draw order: sample the complete ISL vector before OSL.
    input_lengths = _sample_synthetic_lengths(isl, request_count, random_range_ratio, length_rng)
    output_lengths = _sample_synthetic_lengths(osl, request_count, random_range_ratio, length_rng)
    requests = [
        {
            "id": f"synthetic-{index}",
            "arrival_time_ms": arrival_times[index],
            "input_tokens": input_lengths[index],
            "output_tokens": output_lengths[index],
            "metadata": None,
        }
        for index in range(request_count)
    ]
    cached_prefix_tokens = workload.get("cached_prefix_tokens", 0)
    if (
        not isinstance(cached_prefix_tokens, int)
        or isinstance(cached_prefix_tokens, bool)
        or cached_prefix_tokens < 0
        or cached_prefix_tokens > min(input_lengths)
    ):
        raise ValueError(
            "cached_prefix_tokens must be a non-negative integer no greater than every synthetic input length"
        )
    if cached_prefix_tokens:
        prefix = list(range(1, cached_prefix_tokens + 1))
        for index, request in enumerate(requests):
            suffix_length = input_lengths[index] - cached_prefix_tokens
            suffix_seed = (index + 1) * 1_000_003 + cached_prefix_tokens
            request["input_token_ids"] = prefix + [
                (suffix_seed + offset) & 0xFFFF_FFFF for offset in range(suffix_length)
            ]
    return requests, concurrency


def _materialize_sla(spec: ReplaySpec) -> dict[str, JSONValue]:
    raw = spec.goal.get("sla")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("goal.sla must be a mapping")
    return {
        key: _positive_number(raw[key], f"sla.{key}")
        for key in ("ttft_ms", "itl_ms", "e2e_ms")
        if raw.get(key) is not None
    }


def _materialize_engine_role(
    deployment_backend: str,
    deployment_backend_version: str,
    parallel_config: Mapping[str, JSONValue],
    raw_config: Mapping[str, JSONValue],
    role: str,
    *,
    memory_diagnostics: dict[str, Any] | None = None,
) -> dict[str, JSONValue]:
    """Materialize one single-rank or attention-DP generalized engine."""

    role_config = dict(raw_config)
    # The shared CLI/Sweeper form is flat. Nested rank descriptors are already
    # execution-level input and retain the native runtime's compatibility
    # fallback after their structure has been validated below.
    role_memory: dict[str, Any] | None = None
    if memory_diagnostics is not None:
        role_memory = {
            "scope": "capacity_estimate_per_rank",
            "stage": "before_native_capacity_adjustments",
            "status": "unavailable",
            "unavailable_reason": (
                "explicit KV blocks, nested rank input, or a non-AIC capacity provider; "
                "no memory component estimate was used by the Python materializer"
            ),
        }
        memory_diagnostics[role] = role_memory
    capacity_materialized = False
    num_gpu_blocks_is_explicit = False
    if "rank" not in role_config:
        state_cache = _manual_state_cache(role_config, deployment_backend, role)
        num_gpu_blocks_is_explicit = role_config.get("num_gpu_blocks") is not None
        for name in ("aic_backend_version", "backend_version"):
            if role_config.get(name) is None:
                role_config.pop(name, None)
        # Propagate authored versions before capacity preflight. Otherwise a
        # deployment pin reaches timing only after capacity has used current.
        timing = role_config.get("timing_model")
        timing_config = None
        if (
            isinstance(timing, dict)
            and timing.get("type") == "external"
            and timing.get("provider") == "aic"
            and isinstance(timing.get("config"), dict)
        ):
            timing_config = timing["config"]
        version = next(
            (
                value
                for value in (
                    role_config.get("aic_backend_version"),
                    role_config.get("backend_version"),
                    timing_config.get("backend_version") if timing_config is not None else None,
                    deployment_backend_version or None,
                )
                if value is not None
            ),
            None,
        )
        if version is not None:
            if (
                not num_gpu_blocks_is_explicit
                and role_config.get("aic_backend") is not None
                and not any(name in role_config for name in ("aic_backend_version", "backend_version"))
            ):
                role_config["aic_backend_version"] = version
            if timing_config is not None and timing_config.get("backend_version") is None:
                role_config["timing_model"] = {**timing, "config": {**timing_config, "backend_version": version}}
        if state_cache is not None:
            role_config["state_cache"] = state_cache.model_dump(mode="json", exclude_none=True)
        else:
            role_config = materialize_aic_num_gpu_blocks(
                role_config,
                **({"memory_diagnostics": role_memory} if role_memory is not None else {}),
            )
            if role_memory is not None and "total_gpu_capacity_bytes" in role_memory:
                role_memory["status"] = "available"
                role_memory["estimated_num_gpu_blocks"] = role_memory.pop("num_gpu_blocks")
                role_memory.pop("unavailable_reason", None)
            capacity_materialized = role_config.get("num_gpu_blocks") is not None
    for name in ("engine_type", "aic_backend"):
        configured = role_config.pop(name, None)
        if configured is not None and configured != deployment_backend:
            raise ValueError(
                f"{role} {name}={configured!r} conflicts with BackendDeploymentSpec backend={deployment_backend!r}"
            )
    role_config.pop("worker_type", None)
    role_config.pop("startup_time", None)
    model = role_config.pop("aic_model_path", None)
    system = role_config.pop("aic_system", None)
    # Capacity-only input. It has already been consumed by the Python AIC
    # materializer and is not a native scheduler rank field. Preserve it in
    # the AIC timing config because the native runtime rematerializes inferred
    # capacity before execution.
    cuda_graph_reserved_bytes = role_config.pop("cuda_graph_reserved_bytes", None)
    timing_rank = role_config.get("rank", role_config)
    timing = timing_rank.get("timing_model") if isinstance(timing_rank, dict) else None
    if isinstance(timing, dict) and timing.get("type") == "external" and timing.get("provider") == "aic":
        canonical = timing.get("config")
        if not isinstance(canonical, dict):
            raise ValueError("external AIC timing config must be a mapping")
        for aliases, fields in (
            (("tensor_parallel_size", "aic_tp_size"), ("tp", "tp_size")),
            (("dp_size", "aic_attention_dp_size"), ("attention_dp", "attention_dp_size")),
        ):
            expected = canonical.get(fields[0], canonical.get(fields[1]))
            if expected is not None:
                for alias in aliases:
                    if alias in role_config and role_config[alias] != expected:
                        raise ValueError(f"{alias} conflicts with canonical AIC timing topology")
                if not any(alias in role_config for alias in aliases):
                    role_config[aliases[0]] = expected
    raw_dp_size = _pop_matching_aliases(role_config, "attention DP", ("dp_size", "aic_attention_dp_size"), 1)
    raw_tp_size = _pop_matching_aliases(role_config, "tensor parallel", ("tensor_parallel_size", "aic_tp_size"), 1)
    dp_size = _positive_int(
        raw_dp_size,
        f"engine provider {role} dp_size",
    )
    tensor_parallel_size = _positive_int(
        raw_tp_size,
        f"engine provider {role} tensor_parallel_size",
    )
    parallel_prefix = "" if role == "aggregated" else f"{role}_"
    _require_parallel_match(
        parallel_config,
        f"{parallel_prefix}tp",
        tensor_parallel_size,
        f"{role} tensor parallel size",
    )
    _require_parallel_match(
        parallel_config,
        f"{parallel_prefix}attention_dp",
        dp_size,
        f"{role} attention DP size",
    )

    nested_rank = role_config.pop("rank", None)
    if nested_rank is not None:
        if role_config:
            unexpected = ", ".join(sorted(role_config))
            raise ValueError(f"engine provider {role} config cannot mix nested 'rank' with rank fields: {unexpected}")
        if not isinstance(nested_rank, dict):
            raise ValueError(f"engine provider {role} rank config must be a mapping")
        rank: dict[str, JSONValue] = dict(nested_rank)
        num_gpu_blocks_is_explicit = rank.get("num_gpu_blocks") is not None
    else:
        # Preserve the existing convenient flat aggregated form. Disaggregated
        # role mappings may use the same shorthand.
        rank = role_config

    state_cache = _manual_state_cache(rank, deployment_backend, role)
    if state_cache is not None:
        num_gpu_blocks_is_explicit = True
        rank["state_cache"] = state_cache.model_dump(mode="json", exclude_none=True)

    nested_cuda_graph_reserved_bytes = rank.pop("cuda_graph_reserved_bytes", None)
    if cuda_graph_reserved_bytes is not None and nested_cuda_graph_reserved_bytes is not None:
        raise ValueError(f"engine provider {role} config duplicates cuda_graph_reserved_bytes")
    if nested_cuda_graph_reserved_bytes is not None:
        cuda_graph_reserved_bytes = nested_cuda_graph_reserved_bytes
    if cuda_graph_reserved_bytes is not None and (
        not isinstance(cuda_graph_reserved_bytes, int)
        or isinstance(cuda_graph_reserved_bytes, bool)
        or not 0 <= cuda_graph_reserved_bytes <= 1 << 53
    ):
        raise ValueError(
            f"engine provider {role} cuda_graph_reserved_bytes must be a non-negative integer no greater than 2**53"
        )

    configured_backend = rank.get("backend")
    if configured_backend is not None and not isinstance(configured_backend, str):
        raise ValueError(f"engine provider {role} rank backend must be a string")
    backend = configured_backend or deployment_backend
    if backend not in {"vllm", "sglang", "trtllm"}:
        raise ValueError(f"engine replay supports vllm, sglang, and trtllm, got {backend!r}")
    if configured_backend is not None and configured_backend != deployment_backend:
        raise ValueError(
            "engine provider rank backend conflicts with deployment backend: "
            f"{configured_backend!r} != {deployment_backend!r}"
        )
    if "block_size" not in rank and state_cache is None:
        # Keep scheduler capacity, AIC compilation, and synthetic-prefix
        # materialization on one explicit backend-native block size.
        rank["block_size"] = {
            "vllm": 64,
            "sglang": 1,
            "trtllm": 32,
        }[backend]
    rank["backend"] = backend

    memory_fraction_overrides: dict[str, JSONValue] = {}
    for memory_field in (
        "gpu_memory_utilization",
        "mem_fraction_static",
        "free_gpu_memory_fraction",
    ):
        if memory_field not in rank:
            continue
        value = rank.pop(memory_field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"engine provider {role} {memory_field} must be between 0 and 1")
        # Preserve every inferred-capacity input for native rematerialization.
        # A flat config is first materialized in Python, but the Rust runtime
        # repeats that estimate because num_gpu_blocks_is_explicit remains false.
        if not num_gpu_blocks_is_explicit:
            memory_fraction_overrides[memory_field] = float(value)

    aic_timing_overrides = _pop_aic_timing_overrides(rank, role)

    timing_model = rank.get("timing_model")
    uses_aic_timing = timing_model is None or (
        isinstance(timing_model, dict)
        and timing_model.get("type") == "external"
        and timing_model.get("provider") == "aic"
    )
    if cuda_graph_reserved_bytes is not None and uses_aic_timing:
        aic_timing_overrides["cuda_graph_reserved_bytes"] = cuda_graph_reserved_bytes
    elif cuda_graph_reserved_bytes is not None and not capacity_materialized and not num_gpu_blocks_is_explicit:
        raise ValueError(
            f"engine provider {role} cuda_graph_reserved_bytes requires an AIC "
            "timing model when nested rank capacity is inferred; set "
            "rank.num_gpu_blocks explicitly or use AIC timing"
        )
    if deployment_backend_version:

        def resolved_version(value):
            if value not in {"current", "previous", "next"}:
                return value
            from aisimulate_core.sdk.perf_database import resolve_query_version

            from .sweeper.forward_pass_estimator import resolve_systems_paths

            identity = timing_model.get("config", {}) if isinstance(timing_model, dict) else {}
            roots = identity.get("systems_paths")
            if roots is None and identity.get("systems_path") is not None:
                roots = [identity["systems_path"]]
            if roots is None and aic_timing_overrides.get("systems_path") is not None:
                roots = [aic_timing_overrides["systems_path"]]
            return resolve_query_version(
                identity.get("system", system),
                backend,
                value,
                systems_paths=list(resolve_systems_paths(roots)),
            )

        if uses_aic_timing:
            deployment_backend_version = resolved_version(deployment_backend_version)
        configured_version = aic_timing_overrides.get("backend_version")
        if uses_aic_timing:
            configured_version = resolved_version(configured_version)
        if configured_version is not None and configured_version != deployment_backend_version:
            raise ValueError(
                f"engine provider {role} backend version {configured_version!r} "
                "conflicts with BackendDeploymentSpec backend_version="
                f"{deployment_backend_version!r}"
            )
        if uses_aic_timing:
            timing_config = timing_model.get("config") if isinstance(timing_model, dict) else None
            timing_backend_version = timing_config.get("backend_version") if isinstance(timing_config, dict) else None
            timing_backend_version = resolved_version(timing_backend_version)
            if timing_backend_version is not None:
                timing_model = {**timing_model, "config": {**timing_config, "backend_version": timing_backend_version}}
                rank["timing_model"] = timing_model
            if timing_backend_version is not None and timing_backend_version != deployment_backend_version:
                raise ValueError(
                    f"engine provider {role} timing_model.config.backend_version="
                    f"{timing_backend_version!r} conflicts with "
                    "BackendDeploymentSpec backend_version="
                    f"{deployment_backend_version!r}"
                )
            aic_timing_overrides["backend_version"] = deployment_backend_version

    # Identity and capacity inputs may coexist with a fixed/polynomial timing
    # model. They have already served their non-timing purposes and must not be
    # interpreted as an attempt to override that concrete timing model.
    if not uses_aic_timing:
        if any(
            is_active_engine_model_control(name, aic_timing_overrides.get(name)) for name in ENGINE_MODEL_CONTROL_FIELDS
        ):
            raise ValueError("engine model controls require an AIC timing model")
        aic_timing_overrides.clear()
        if capacity_materialized:
            memory_fraction_overrides.clear()

    speculation_raw = rank.pop("speculation", None)
    speculation = None
    if speculation_raw is not None:
        from .config.engine import NgramSpeculationConfig

        speculation = NgramSpeculationConfig.model_validate(speculation_raw)
        if backend != "vllm" or rank.get("native_host_offload") is not None:
            raise ValueError("ngram speculation requires vllm without host_offload")
        if any(
            rank.get(key) is not None
            for key in (
                "aic_nextn",
                "nextn",
                "aic_nextn_accept_rates",
                "nextn_accept_rates",
                "aic_mtp_seed",
                "mtp_seed",
            )
        ):
            raise ValueError("speculation cannot be combined with legacy speculative decoding fields")
        rank["aic_nextn"] = speculation.num_speculative_tokens
        rank["aic_nextn_accept_rates"] = ",".join(str(rate) for rate in speculation.acceptance_rates)
        rank["aic_mtp_seed"] = speculation.seed

    nextn = _pop_alias(rank, "aic_nextn", ("aic_nextn", "nextn"))
    if nextn is not None:
        nextn = _positive_int(nextn, f"engine provider {role} aic_nextn")
        if nextn > 5:
            raise ValueError(f"engine provider {role} aic_nextn must be in 1..=5")
        rank["aic_nextn"] = nextn

    accept_rates = _pop_alias(
        rank,
        "aic_nextn_accept_rates",
        ("aic_nextn_accept_rates", "nextn_accept_rates"),
    )
    if accept_rates is not None:
        if nextn is None:
            raise ValueError(f"engine provider {role} aic_nextn_accept_rates requires aic_nextn")
        if not isinstance(accept_rates, str):
            raise ValueError(f"engine provider {role} aic_nextn_accept_rates must be a string")
        rank["aic_nextn_accept_rates"] = accept_rates

    nextn_accepted = _pop_alias(
        rank,
        "aic_nextn_accepted",
        ("aic_nextn_accepted", "nextn_accepted"),
    )
    if nextn_accepted is not None:
        if nextn is None:
            raise ValueError(f"engine provider {role} aic_nextn_accepted requires aic_nextn")
        if accept_rates is not None:
            raise ValueError(f"engine provider {role} cannot set both aic_nextn_accepted and aic_nextn_accept_rates")
        rank["aic_nextn_accept_rates"] = _accept_rates_for_expected(nextn, nextn_accepted, role=role)

    mtp_seed = _pop_alias(rank, "aic_mtp_seed", ("aic_mtp_seed", "mtp_seed"))
    if mtp_seed is not None:
        if not isinstance(mtp_seed, int) or isinstance(mtp_seed, bool) or not 0 <= mtp_seed <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(f"engine provider {role} aic_mtp_seed must be an unsigned 64-bit integer")
        rank["aic_mtp_seed"] = mtp_seed

    if "timing_model" not in rank:
        if not isinstance(model, str) or not model:
            raise ValueError(f"{role} engine arguments require aic_model_path")
        if not isinstance(system, str) or not system:
            raise ValueError(f"{role} engine arguments require aic_system")
        timing_config: dict[str, JSONValue] = {
            "model": model,
            "backend": backend,
            "system": system,
            "tp": tensor_parallel_size,
            "attention_dp": dp_size,
        }
        block_size = rank.get("block_size")
        if isinstance(block_size, int) and not isinstance(block_size, bool):
            timing_config["kv_block_size"] = block_size
        timing_config.update(memory_fraction_overrides)
        timing_config.update(aic_timing_overrides)
        if nextn is not None and speculation is None:
            timing_config["nextn"] = nextn
        rank["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": timing_config,
        }
    elif memory_fraction_overrides or aic_timing_overrides:
        timing_model = rank["timing_model"]
        if (
            not isinstance(timing_model, dict)
            or timing_model.get("type") != "external"
            or timing_model.get("provider") != "aic"
            or not isinstance(timing_model.get("config"), dict)
        ):
            raise ValueError("engine AIC overrides require an AIC timing model")
        timing_model = dict(timing_model)
        timing_config = dict(timing_model["config"])
        timing_overrides = memory_fraction_overrides | aic_timing_overrides
        duplicates = timing_overrides.keys() & timing_config.keys()
        if "backend_version" in duplicates and timing_overrides["backend_version"] == timing_config["backend_version"]:
            duplicates.remove("backend_version")
        if duplicates:
            duplicate = ", ".join(sorted(duplicates))
            raise ValueError(
                f"engine AIC option is configured both on the rank and inside timing_model.config: {duplicate}"
            )
        timing_config.update(timing_overrides)
        timing_model["config"] = timing_config
        rank["timing_model"] = timing_model

    if nextn is not None:
        timing_model = rank["timing_model"]
        if (
            isinstance(timing_model, dict)
            and timing_model.get("type") == "external"
            and timing_model.get("provider") == "aic"
            and isinstance(timing_model.get("config"), dict)
        ):
            timing_model = dict(timing_model)
            timing_config = dict(timing_model["config"])
            if speculation is not None:
                cost_config = speculation.cost_config()
                if timing_config.get("nextn") not in (None, 0):
                    raise ValueError("ngram speculation conflicts with timing_model.config.nextn")
                if timing_config.get("speculation") not in (None, cost_config):
                    raise ValueError("ngram speculation conflicts with timing_model.config.speculation")
                if timing_config.get("forward_model", "op_level") != "op_level":
                    raise ValueError("ngram speculation requires op_level timing")
                timing_config["speculation"] = cost_config
            configured_nextn = timing_config.get("nextn")
            if speculation is None and configured_nextn is not None and configured_nextn != nextn:
                raise ValueError(
                    f"engine provider {role} aic_nextn={nextn} conflicts with "
                    f"timing_model.config.nextn={configured_nextn!r}"
                )
            if speculation is None:
                timing_config["nextn"] = nextn
            timing_model["config"] = timing_config
            rank["timing_model"] = timing_model

    return {
        "dp_size": dp_size,
        "tensor_parallel_size": tensor_parallel_size,
        "num_gpu_blocks_is_explicit": num_gpu_blocks_is_explicit,
        "rank": rank,
    }


def _manual_state_cache(rank: Mapping[str, JSONValue], backend: str, role: str) -> StateCacheConfig | None:
    """Validate manual geometry before any automatic capacity materialization."""

    raw = rank.get("state_cache")
    if raw is None:
        return None
    state_cache = StateCacheConfig.model_validate(raw)
    if rank.get("prefix_match_unit") is not None and rank.get("aic_nextn") is not None:
        raise ValueError("prefix_match_unit does not support speculative decoding")
    if backend != "vllm" or role != "aggregated":
        raise ValueError("state_cache requires backend=vllm and an aggregated G1 worker")
    if rank.get("native_host_offload") is not None:
        raise ValueError("state_cache supports G1 only; native_host_offload is not supported")
    if rank.get("g3_offload") is not None:
        raise ValueError("state_cache supports G1 only; g3_offload is not supported")
    conflicts = [
        name
        for name in (
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "kv_transfer_bytes_per_token",
            "kv_transfer_bandwidth",
        )
        if rank.get(name) is not None
    ]
    if rank.get("cuda_graph_reserved_bytes") not in (None, 0):
        conflicts.append("cuda_graph_reserved_bytes")
    # Rust serializes this default explicitly; it does not enable PD transfer.
    if rank.get("kv_transfer_timing_mode") not in (None, "full_prompt"):
        conflicts.append("kv_transfer_timing_mode")
    if conflicts:
        raise ValueError(f"state_cache rejects capacity/transfer overrides: {', '.join(conflicts)}")
    for name in ("num_gpu_blocks", "block_size", "kv_cache_bytes_per_token"):
        value = rank.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= (1 << 64) - 1:
            raise ValueError(f"state_cache requires explicit positive {name} within u64")
    if rank["num_gpu_blocks"] < state_cache.state_blocks(rank["block_size"], rank["kv_cache_bytes_per_token"]) + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return state_cache


def _positive_int(value: JSONValue, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _accept_rates_for_expected(nextn: int, value: JSONValue, *, role: str) -> str:
    """Lower an explicit expected accepted-token count to conditional rates."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= nextn
    ):
        raise ValueError(f"engine provider {role} aic_nextn_accepted must be finite and within [0, {nextn}]")
    expected = float(value)
    whole = int(expected)
    fraction = expected - whole
    rates = [1.0] * whole
    if len(rates) < nextn:
        rates.append(fraction)
    rates.extend([0.0] * (nextn - len(rates)))
    return ",".join(format(rate, ".17g") for rate in rates)


def _random_range_ratio(value: JSONValue) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
        or value > 1.0
    ):
        raise ValueError(f"random_range_ratio must be finite and in (0.0, 1.0], got {value!r}")
    return float(value)


def _random_seed(value: JSONValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("random_seed must be an unsigned 64-bit integer")
    return value


def _sample_synthetic_lengths(
    upper: int,
    count: int,
    random_range_ratio: float,
    rng: random.Random | np.random.RandomState,
) -> list[int]:
    if random_range_ratio == 1.0:
        return [upper] * count
    lower = int(upper * random_range_ratio)
    if lower == 0:
        raise ValueError(f"random_range_ratio={random_range_ratio} gives a zero-token lower bound for length {upper}")
    if isinstance(rng, np.random.RandomState):
        return rng.randint(lower, upper + 1, size=count).tolist()
    return [rng.randint(lower, upper) for _ in range(count)]


def _require_parallel_match(
    parallel_config: Mapping[str, JSONValue],
    field: str,
    actual: int,
    label: str,
) -> None:
    expected = parallel_config.get(field)
    if expected is None:
        return
    expected = _positive_int(expected, f"parallel_config.{field}")
    if expected != actual:
        raise ValueError(f"parallel_config.{field}={expected} conflicts with {label}={actual}")


def _require_supported_synthetic_workload(
    workload: Mapping[str, JSONValue],
) -> None:
    unsupported: list[str] = []
    if workload.get("turns_per_session", 1) != 1:
        unsupported.append("turns_per_session")
    if workload.get("shared_prefix_ratio", 0.0) != 0.0:
        unsupported.append("shared_prefix_ratio")
    if workload.get("num_prefix_groups", 0) != 0:
        unsupported.append("num_prefix_groups")
    if workload.get("inter_turn_delay_ms", 0.0) != 0.0:
        unsupported.append("inter_turn_delay_ms")
    if unsupported:
        raise ValueError("engine replay synthetic traffic does not yet support " + ", ".join(unsupported))


def _pop_alias(
    config: dict[str, JSONValue],
    target: str,
    aliases: tuple[str, ...],
) -> JSONValue | None:
    configured = [alias for alias in aliases if alias in config]
    if len(configured) > 1:
        names = ", ".join(configured)
        raise ValueError(f"engine config duplicates {target}: {names}")
    if not configured:
        return None
    return config.pop(configured[0])


def _pop_matching_aliases(
    config: dict[str, JSONValue],
    target: str,
    aliases: tuple[str, ...],
    default: JSONValue,
) -> JSONValue:
    values = [config.pop(alias) for alias in aliases if alias in config]
    if not values:
        return default
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"engine config has conflicting {target} aliases")
    return values[0]


def _nonnegative_time(value: JSONValue, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive_number(value: JSONValue, name: str) -> float:
    number = _nonnegative_time(value, name)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _normalize_engine_replay_report(report: Mapping[str, JSONValue], *, include_native_report: bool) -> ReplayReport:
    """Normalize an execution report to Sweeper's stable scoring metric names."""

    payload = dict(report)
    metrics: dict[str, float | None] = {}
    try:
        power = normalize_power_summary(payload)
    except ValueError as exc:
        raise InvalidRunnerError(str(exc)) from exc
    payload.update(power)

    def add(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidRunnerError(f"engine replay metric {name!r} is not finite") from exc
        if not math.isfinite(number):
            raise InvalidRunnerError(f"engine replay metric {name!r} is not finite")
        metrics[name] = number

    # The Rust ReplayReport's flat serialization is the sole wire
    # format. Per-request records remain a Rust replay concern and are not
    # reconstructed into a second Python report model.
    for name, value in payload.items():
        add(name, value)
    metadata = {
        key: payload[key]
        for key in (
            "agentic_qualification",
            "agentic_input_format",
            "agentic_lanes",
            "agentic_snapshots",
            "agentic_phases",
            "agentic_model_projection",
            "weka_nested_timestamp_basis",
        )
        if key in payload
    }
    if include_native_report:
        metadata["native_report"] = payload
    metrics.update(normalize_power_summary(metrics))
    metadata["power"] = power_metadata(metrics)
    return ReplayReport(metrics=metrics, metadata=metadata)
