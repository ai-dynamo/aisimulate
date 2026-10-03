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
from dataclasses import dataclass
from typing import Any

from .compiler import prediction_to_replay_spec
from .config.cli import CorePredictionConfig
from .config_adapter import PredictionAdapterContext, SimulationConfigAdapter
from .detail import prediction_summary
from .power import normalize_power_summary
from .resources import ResourceLimitError
from .sweeper.provider import AdapterReplaySpec
from .sweeper.replay import ReplayOutputRequirements, ReplayReport, ReplaySpec, RunnerFactory


class PredictionExecutionError(RuntimeError):
    """A runner failure while executing a prediction replay."""


@dataclass(frozen=True)
class PredictionResult:
    """Structured result of :func:`run_prediction`.

    Attributes:
        summary: The merged prediction summary (throughput/latency/power
            metrics), the same mapping the CLI renders.
        native: The full native report with ``summary`` merged in.
        replay_spec: The compiled replay spec that was executed (e.g. for AFD
            qualification artifacts).
        report: The raw :class:`ReplayReport` returned by the runner.
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
    runner = runner_factory.create(0)
    try:
        try:
            report = runner.run(spec, output_requirements=output_requirements)
        except (KeyboardInterrupt, ResourceLimitError):
            # ResourceLimitError from a guarded runner must reach the caller
            # unwrapped so it can be handled distinctly from execution failures.
            raise
        except Exception as exc:
            raise PredictionExecutionError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        runner.close()
    native = report.metadata.get("native_report")
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
    summary.update(normalize_power_summary(report.metrics))
    if "summary" in native:
        native = {**native, "summary": summary}
    else:
        native = {**native, **summary}
    return PredictionResult(summary=summary, native=native, replay_spec=spec, report=report)
