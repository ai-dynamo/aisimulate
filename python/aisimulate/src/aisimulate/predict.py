# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute the public prediction configuration.

Library entry point for a single-point prediction, symmetric to
:func:`aisimulate.recommend.run_recommendation`. It compiles a
:class:`CorePredictionConfig` into a replay spec, runs it on the supplied
runner, and returns the structured summary and native report. CLI-only concerns
(output directory, report files, stdout, detail rendering) stay with the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .compiler import prediction_to_replay_spec
from .config.cli import CorePredictionConfig
from .config_adapter import PredictionAdapterContext, SimulationConfigAdapter
from .detail import SERVING_LATENCY_METRICS, SERVING_THROUGHPUT_METRICS, prediction_summary
from .power import normalize_power_summary
from .resources import ResourceLimitError
from .sweeper.provider import AdapterReplaySpec
from .sweeper.replay import ReplayOutputRequirements, ReplayReport, ReplaySpec, RunnerFactory


class PredictionExecutionError(RuntimeError):
    """A runner failure while executing a prediction replay."""

    def __init__(self, message: str, *, fpm_query_coverage: Any = None) -> None:
        super().__init__(message)
        self.fpm_query_coverage = fpm_query_coverage


@dataclass(frozen=True)
class PredictionResult:
    """Structured result of :func:`run_prediction`.

    Attributes:
        summary: The merged prediction summary (throughput/latency/power
            metrics), the same mapping the CLI renders.
        native: The full native report with ``summary`` merged in.
        replay_spec: The compiled replay spec that was executed (e.g. for AFD
            qualification artifacts).
        report: The runner's :class:`ReplayReport`, retaining requested diagnostics.
    """

    summary: dict[str, Any]
    native: dict[str, Any]
    replay_spec: ReplaySpec
    report: ReplayReport


def _prediction_adapter_context(config: CorePredictionConfig) -> PredictionAdapterContext:
    return PredictionAdapterContext(
        engine=config.engine.model_dump(mode="json", exclude_none=True),
        traffic=config.traffic.model_dump(mode="json", exclude_none=True),
        evaluation=config.evaluation.model_dump(mode="json", exclude_none=True),
    )


def _compile_prediction_adapters(
    configs: Mapping[str, Mapping[str, Any]],
    adapters: Mapping[str, SimulationConfigAdapter],
    *,
    stack: str,
    context: PredictionAdapterContext,
) -> dict[str, AdapterReplaySpec]:
    compiled: dict[str, AdapterReplaySpec] = {}
    for section, raw in configs.items():
        name = f"{stack}.{section}"
        adapter = adapters[name]
        compiled[name] = adapter.compile_prediction(raw, context)
    return compiled


def run_prediction(
    config: CorePredictionConfig,
    *,
    adapter_configs: Mapping[str, Mapping[str, Any]] | None = None,
    stack: str,
    runner_factory: RunnerFactory,
    providers: Mapping[str, SimulationConfigAdapter] | None = None,
    execution_mode: str = "offline",
    output_requirements: ReplayOutputRequirements | None = None,
) -> PredictionResult:
    """Run a public prediction through the replay runner core.

    Args:
        config: The validated public prediction configuration.
        adapter_configs: Raw per-section adapter configuration blocks, keyed by
            section name (the same shape the CLI splits out of the config file).
        stack: The runner stack name used to namespace adapter sections.
        runner_factory: Factory that creates the replay runner and reports its
            capabilities.
        providers: Resolved config adapters keyed by ``"<stack>.<section>"``.
        execution_mode: ``"offline"`` (default) or ``"online"``.
        output_requirements: Optional replay output requirements. When omitted,
            defaults to raw-report capture for non-EPD predictions.

    Returns:
        A :class:`PredictionResult` with the summary, native report, replay
        spec, and raw runner report.

    Raises:
        PredictionExecutionError: If the runner fails while executing the spec.
    """
    adapter_configs = dict(adapter_configs or {})
    providers = dict(providers or {})
    epd = config.engine.workers.encoder is not None
    adapter_specs = _compile_prediction_adapters(
        adapter_configs,
        providers,
        stack=stack,
        context=_prediction_adapter_context(config),
    )
    spec = prediction_to_replay_spec(
        config,
        adapter_specs=adapter_specs,
        execution_mode=execution_mode,
    )
    runner_factory.capabilities().require_compatible(spec)
    if output_requirements is None:
        output_requirements = ReplayOutputRequirements(include_raw_report=not epd)
    # Capacity is already computed by the runner. Retain its resolved precision
    # for ordinary output too, without recomputing a model or its memory budget.
    capture_memory_details = output_requirements.capture_memory_diagnostics
    retain_native_report = (
        output_requirements.include_raw_report
        or output_requirements.capture_per_request
        or capture_memory_details
        or output_requirements.capture_performance_diagnostics
    )
    output_requirements = replace(output_requirements, capture_memory_diagnostics=True)
    runner = runner_factory.create(0)
    try:
        try:
            report = runner.run(spec, output_requirements=output_requirements)
        except (KeyboardInterrupt, ResourceLimitError):
            # ResourceLimitError from a guarded runner must reach the caller
            # unwrapped so it can be handled distinctly from execution failures.
            raise
        except Exception as exc:
            raise PredictionExecutionError(
                f"{type(exc).__name__}: {exc}",
                fpm_query_coverage=getattr(exc, "fpm_query_coverage", None),
            ) from exc
    finally:
        runner.close()
    native = report.metadata.get("native_report")
    memory_diagnostics = (
        native.get("memory_diagnostics", {})
        if isinstance(native, dict)
        else report.metadata.get("memory_diagnostics", {})
    )
    metadata = dict(report.metadata)
    if not capture_memory_details:
        metadata.pop("memory_diagnostics", None)
        if isinstance(native, dict):
            native = {key: value for key, value in native.items() if key != "memory_diagnostics"}
            metadata["native_report"] = native
    if not retain_native_report:
        metadata.pop("native_report", None)
        native = None
    report = replace(report, metadata=metadata)
    if not isinstance(native, dict):
        native = {"summary": dict(report.metrics)}
    if epd:
        native = {"summary": dict(report.metrics), "metadata": dict(report.metadata)}
        if "memory_diagnostics" in native["metadata"]:
            native["memory_diagnostics"] = native["metadata"].pop("memory_diagnostics")
        # JSON stdout, like prediction.json, must identify the approximation.
        missing = [key for key in ("metric_semantics", "total_gpus") if key not in report.metadata]
        if missing:
            raise PredictionExecutionError(
                f"analytical EPD report is missing required metadata field(s): {', '.join(missing)}"
            )
        native["summary"]["metric_semantics"] = report.metadata["metric_semantics"]
        native["summary"]["total_gpus"] = report.metadata["total_gpus"]
    summary = prediction_summary(native)
    kv_precision = {}
    for role, memory in memory_diagnostics.items():
        mode = memory.get("kvcache_quant_mode")
        worker = getattr(config.engine.workers, role, None)
        if mode is None or worker is None:
            continue
        explicit = worker.timing.kvcache_quant_mode or config.engine.kvcache_quant_mode
        if explicit is not None:
            source = "user_override"
        elif memory.get("source") == "profile":
            source = "fpm_profile"
        else:
            # Reuse the cached, normalized checkpoint metadata read during model
            # construction. Weight quantization alone is not a KV declaration.
            raw = get_model_config_from_model_path(config.engine.model).get("raw_config", {})
            declared = raw.get("kv_cache_quant_algo")
            declared = "bfloat16" if declared == "none" else declared
            source = "checkpoint" if declared == mode else "aisimulate_policy"
        kv_precision[role] = {"dtype": mode, "source": source}
    if kv_precision:
        summary["kv_cache_precision"] = kv_precision
    provenance = report.metadata.get("prediction_provenance")
    if provenance is None or provenance == {}:
        provenance = {"runner": {"unavailable_reason": "runner did not report effective timing provenance"}}
    if not isinstance(provenance, dict):
        raise PredictionExecutionError("runner prediction_provenance must be a mapping of worker roles")
    for role, entry in provenance.items():
        if not isinstance(role, str) or not role or not isinstance(entry, dict):
            raise PredictionExecutionError("runner prediction_provenance requires named roles with mapping values")
        if entry.get("unavailable_reason"):
            required = ("unavailable_reason",)
        else:
            version_key = "data_version" if entry.get("data_version") is not None else "data_version_unavailable_reason"
            required = ("estimator", "timing_source", version_key)
        if any(not isinstance(entry.get(key), str) or not entry[key].strip() for key in required):
            raise PredictionExecutionError(
                f"runner prediction_provenance.{role} requires nonempty strings for {required}"
            )
    summary["prediction_provenance"] = provenance
    summary.update(normalize_power_summary(report.metrics))
    if summary.get("completed_requests") == 0:
        summary.update(
            prediction_status="invalid",
            stop_reason="no_completed_requests",
            diagnostic=(
                "No requests completed; serving latency and throughput are unavailable. "
                "Check KV capacity against request lengths, context limits, and traffic stop limits."
            ),
        )
        for key in summary:
            if key in SERVING_LATENCY_METRICS or key in SERVING_THROUGHPUT_METRICS:
                summary[key] = None
    if "summary" in native:
        native = {**native, "summary": summary}
    else:
        native = {**native, **summary}
    return PredictionResult(summary=summary, native=native, replay_spec=spec, report=report)
