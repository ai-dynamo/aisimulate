# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single public AISimulate command-line application."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from .afd_artifacts import write_afd_qualification_artifacts
from .cli_args import _apply_overrides, _CliConfigError, _load_mapping, build_parser
from .compiler import prediction_to_replay_spec
from .config.cli import (
    CorePredictionConfig,
    CoreRecommendationConfig,
    prediction_mapping,
)
from .config.common import split_config_sections
from .config_adapter import (
    ConfigAdapterResolutionError,
    PredictionAdapterContext,
    SimulationConfigAdapter,
    resolve_config_adapters,
)
from .detail import build_prediction_details, energy_diagnostics, prediction_summary
from .output import (
    format_prediction_stdout,
    format_recommendation_stdout,
    prepare_output_directory,
    write_prediction_report,
    write_recommendation_csv,
    write_recommendation_result,
    write_recommendations,
    write_requests,
)
from .power import normalize_power_summary
from .resources import (
    GuardedRunnerFactory,
    ResourceLimitError,
    build_plan,
    require_plan,
    workload_bounds,
)
from .stack import StackResolutionError, resolve_runner_factory
from .sweeper.provider import AdapterReplaySpec
from .sweeper.replay import ReplayOutputRequirements


class _CliExecutionError(RuntimeError):
    pass


def _resolve_section_adapters(sections: dict[str, dict[str, Any]], stack: str) -> dict[str, SimulationConfigAdapter]:
    adapters = resolve_config_adapters(f"{stack}.{section}" for section in sections)
    for name, adapter in adapters.items():
        if adapter.section not in sections:
            raise ConfigAdapterResolutionError(f"config adapter {name!r} does not match a configured section")
    return adapters


def _prediction_adapter_context(
    config: CorePredictionConfig,
) -> PredictionAdapterContext:
    return PredictionAdapterContext(
        engine=config.engine.model_dump(mode="json", exclude_none=True),
        traffic=config.traffic.model_dump(mode="json", exclude_none=True),
        evaluation=config.evaluation.model_dump(mode="json", exclude_none=True),
    )


def _compile_prediction_adapters(
    configs: dict[str, dict[str, Any]],
    adapters: dict[str, SimulationConfigAdapter],
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


def _predict(args: argparse.Namespace, raw: dict[str, Any], factory) -> int:
    core_raw, adapter_raw = split_config_sections(raw, command="predict")
    config = CorePredictionConfig.model_validate(core_raw)
    if config.engine.speculation is not None and (args.stack != "engine" or args.online or adapter_raw):
        raise ValueError("ngram speculation requires offline --stack engine without adapters")
    plan = _resource_plan(args, config, factory)
    require_plan(plan)
    from .supervision import checkpoint, mark_execution_ready, mark_shutdown

    checkpoint("resource_plan", plan)
    factory = GuardedRunnerFactory(factory, args.stack, config.execution.resources)
    epd = config.engine.workers.encoder is not None
    if epd and (args.stack != "engine" or args.online or args.capture_per_request or adapter_raw):
        raise ValueError("analytical EPD requires offline --stack engine without adapters or per-request capture")
    adapters = _resolve_section_adapters(adapter_raw, args.stack)
    adapter_specs = _compile_prediction_adapters(
        adapter_raw,
        adapters,
        stack=args.stack,
        context=_prediction_adapter_context(config),
    )
    spec = prediction_to_replay_spec(
        config,
        adapter_specs=adapter_specs,
        execution_mode="online" if args.online else "offline",
    )
    factory.capabilities().require_compatible(spec)
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    runner = factory.create(0)
    mark_execution_ready()
    try:
        try:
            report = runner.run(
                spec,
                output_requirements=ReplayOutputRequirements(
                    include_raw_report=not epd,
                    capture_per_request=args.capture_per_request,
                    capture_memory_diagnostics="memory" in args.detail,
                ),
            )
        except (KeyboardInterrupt, ResourceLimitError):
            raise
        except Exception as exc:
            raise _CliExecutionError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        mark_shutdown()
        runner.close()
        mark_execution_ready()
    native = report.metadata.get("native_report")
    if not isinstance(native, dict):
        native = {"summary": dict(report.metrics)}
    if epd:
        native = {"summary": dict(report.metrics), "metadata": dict(report.metadata)}
        if "memory_diagnostics" in native["metadata"]:
            native["memory_diagnostics"] = native["metadata"].pop("memory_diagnostics")
        # JSON stdout, like prediction.json, must identify the approximation.
        native["summary"]["metric_semantics"] = report.metadata["metric_semantics"]
        native["summary"]["total_gpus"] = report.metadata["total_gpus"]
    summary = prediction_summary(native)
    summary.update(normalize_power_summary(report.metrics))
    if "summary" in native:
        native = {**native, "summary": summary}
    else:
        native = {**native, **summary}
    power_diagnostics = None
    if args.diagnostics == "power":
        power_diagnostics = energy_diagnostics(native)
        native = {**native, "power_diagnostics": power_diagnostics}
    resolved_basis = native.get("weka_nested_timestamp_basis")
    if isinstance(resolved_basis, str):
        source = config.traffic.source
        requested_basis = getattr(source, "nested_timestamp_basis", None) or "auto"
        if requested_basis == "auto":
            sys.stderr.write(
                "INFO: heuristically resolved one nested timestamp basis after validating the complete "
                f"Weka corpus: requested='auto', resolved={resolved_basis!r}\n"
            )
        else:
            sys.stderr.write(
                "INFO: validated the complete Weka corpus with configured "
                f"nested_timestamp_basis requested={requested_basis!r}, resolved={resolved_basis!r}\n"
            )
    details = build_prediction_details(native, args.detail) if args.detail else None
    if details is not None:
        native = {**native, "details": details}
    report_path = write_prediction_report(root, native)
    write_afd_qualification_artifacts(root, spec)
    if args.capture_per_request:
        records = native.get("per_request")
        if not isinstance(records, list):
            raise RuntimeError("selected stack did not provide per-request prediction records")
        if any(not isinstance(record, dict) for record in records):
            raise RuntimeError("per-request records must be JSON mappings")
        write_requests(root, records)
    phases = report.metadata.get("agentic_phases")
    if isinstance(phases, dict) and phases.get("phase") == "aborted":
        sys.stderr.write(
            f"ERROR: agentic preparation aborted: {phases.get('failure_reason') or 'preparation did not complete'}; "
            f"saved full report to: {report_path}\n"
        )
        return 1
    sys.stdout.write(
        format_prediction_stdout(
            summary,
            args.format,
            details=details,
            power_diagnostics=power_diagnostics,
            diagnostics_top_n=args.diagnostics_top_n,
        )
    )
    sys.stdout.write("\n")
    if args.format == "table":
        sys.stdout.write(f"Saved full report to: {report_path}\n")
    return 0


def _recommend(args: argparse.Namespace, raw: dict[str, Any], factory) -> int:
    from .recommend import run_recommendation

    core_raw, adapter_raw = split_config_sections(raw, command="recommend")
    config = CoreRecommendationConfig.model_validate(core_raw)
    adapters = _resolve_section_adapters(adapter_raw, args.stack)
    result = run_recommendation(
        config,
        adapter_configs=adapter_raw,
        stack=args.stack,
        runner_factory=factory,
        providers=adapters,
        show_progress=args.format == "table",
    )
    selected: list[tuple[str, Any, dict[str, Any]]] = []
    seen_configs: set[str] = set()
    for candidate_id, candidate in zip(
        result.selected_candidate_ids,
        result.selected_candidates,
        strict=True,
    ):
        if candidate.prediction_config is None:
            raise RuntimeError("recommendation candidate has no concrete public config")
        candidate_core, candidate_adapters = split_config_sections(candidate.prediction_config, command="predict")
        prediction = CorePredictionConfig.model_validate(candidate_core)
        unknown_sections = set(candidate_adapters) - set(adapter_raw)
        if unknown_sections:
            raise RuntimeError(f"recommendation produced unconfigured adapter sections {sorted(unknown_sections)}")
        compiled_adapters = _compile_prediction_adapters(
            candidate_adapters,
            adapters,
            stack=args.stack,
            context=_prediction_adapter_context(prediction),
        )
        concrete = prediction_mapping(
            prediction,
            {adapters[name].section: spec.config for name, spec in compiled_adapters.items()},
        )
        config_key = json.dumps(concrete, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        selected.append((candidate_id, candidate, concrete))
    result = result.with_selected_prediction_configs(
        [(candidate_id, concrete) for candidate_id, _, concrete in selected]
    )
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    result_path = write_recommendation_result(root, result)
    write_recommendation_csv(root, result)
    if not selected:
        sys.stderr.write(f"no feasible candidate found; saved full result to: {result_path}\n")
        return 3 if getattr(result.counts, "resource_limited", 0) else 1
    paths = write_recommendations(root, [config for _, _, config in selected])
    rows = []
    for index, ((_, candidate, _), path) in enumerate(zip(selected, paths, strict=True), start=1):
        row = {
            "rank": index,
            "score": candidate.score,
            "objectives": candidate.objectives,
            "used_gpus": candidate.used_gpus,
            "config_path": str(path),
        }
        row.update(normalize_power_summary(candidate.metrics))
        rows.append(row)
    sys.stdout.write(format_recommendation_stdout(rows, args.format))
    sys.stdout.write("\n")
    if args.format == "table":
        sys.stdout.write(f"Saved full result to: {result_path}\n")
    if getattr(result.counts, "resource_limited", 0):
        sys.stderr.write("some candidates were resource-limited; saved results cover only evaluated candidates\n")
        return 3
    return 0


def _resource_plan(args, config, factory) -> dict[str, Any]:
    return build_plan(
        workload_bounds(config),
        stack=args.stack,
        policy=config.execution.resources,
        requested_parallelism=config.optimizer.parallelism if isinstance(config, CoreRecommendationConfig) else 1,
        factory=factory,
    )


def _write_resource_plan(args, plan: dict[str, Any]) -> None:
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    (root / "resource-plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    # Stack resolution deliberately precedes opening the configuration file.
    try:
        factory = resolve_runner_factory(args.stack)
    except StackResolutionError as exc:
        parser.error(str(exc))
    try:
        raw = _load_mapping(args.config)
        _apply_overrides(raw, args.overrides, command=args.command)
        if args.command == "predict":
            return _predict(args, raw, factory)
        return _recommend(args, raw, factory)
    except (
        _CliConfigError,
        ConfigAdapterResolutionError,
        ValidationError,
        ValueError,
    ) as exc:
        parser.error(f"{args.config}: {exc}")
    except KeyboardInterrupt:
        return 130
    except ResourceLimitError as exc:
        sys.stderr.write(f"aisimulate {args.command}: {exc}\n")
        try:
            _write_resource_plan(args, exc.plan)
        except (OSError, ValueError) as output_error:
            sys.stderr.write(f"could not save resource plan: {output_error}\n")
        return 3
    except _CliExecutionError as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    from .supervision import main as supervised_main

    raise SystemExit(supervised_main())
