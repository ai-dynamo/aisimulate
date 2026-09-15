# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single public AISimulate command-line application."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .afd_artifacts import write_afd_qualification_artifacts
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
from .detail import build_prediction_details, parse_detail_sections, prediction_summary
from .output import (
    format_prediction_stdout,
    format_recommendation_stdout,
    prepare_output_directory,
    write_prediction_report,
    write_recommendation_result,
    write_recommendations,
    write_requests,
)
from .stack import StackResolutionError, resolve_runner_factory
from .sweeper.provider import AdapterReplaySpec
from .sweeper.replay import ReplayOutputRequirements


class _CliConfigError(ValueError):
    pass


class _CliExecutionError(RuntimeError):
    pass


def build_parser(*, estimate_arguments: bool = True) -> argparse.ArgumentParser:
    """Build the CLI; main skips estimator options for unrelated invocations."""
    parser = argparse.ArgumentParser(
        prog="aisimulate",
        description="Predict serving behavior, recommend configurations, or estimate a fixed batch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "recommend"):
        child = subparsers.add_parser(command)
        child.add_argument("-c", "--config", required=True)
        child.add_argument("--stack", default="engine")
        child.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="PATH=YAML_VALUE",
        )
        child.add_argument("--output-dir", default="./aisimulate-output")
        child.add_argument("--overwrite", action="store_true")
        child.add_argument("--format", choices=("table", "json"), default="table")
    estimate = subparsers.add_parser(
        "estimate",
        help="Estimate a fixed configuration or static prefill/decode batch.",
        description="Estimate a fixed configuration with the bundled estimator. "
        "Uses explicit model, hardware, batch, and parallelism flags; no serving-workload YAML is required.",
    )
    if estimate_arguments:
        # Keep estimator imports out of predict/recommend and top-level help.
        from aiconfigurator.cli.main import configure_estimate_parser

        configure_estimate_parser(estimate)
    subparsers.choices["predict"].add_argument("--capture-per-request", action="store_true")
    subparsers.choices["predict"].epilog = (
        "AgentX M1: use traffic.source.format=weka or agentic_mooncake with "
        "trace_timestamps and agentic_lanes=1. The engine stack supports aggregated "
        "vLLM/SGLang, HBM-only, speculative decoding disabled. Results are "
        "functional_only; benchmark warmup and profiling are not qualified."
    )
    subparsers.choices["predict"].add_argument(
        "--detail",
        type=parse_detail_sections,
        default=(),
        metavar="SECTIONS",
        help="comma-separated summary,memory,time, or all; unavailable evidence is skipped",
    )
    subparsers.choices["predict"].add_argument(
        "--online",
        action="store_true",
        help="pace prediction against the real wall clock instead of virtual time",
    )
    return parser


def _load_mapping(path: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _CliConfigError(f"could not read configuration {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise _CliConfigError(f"malformed YAML in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise _CliConfigError(f"configuration {source} must contain one YAML mapping")
    return value


def _apply_overrides(data: dict[str, Any], overrides: list[str], *, command: str) -> None:
    for assignment in overrides:
        if "=" not in assignment:
            raise _CliConfigError(f"invalid --set {assignment!r}; expected PATH=YAML_VALUE")
        raw_path, raw_value = assignment.split("=", 1)
        parts = raw_path.split(".")
        if not raw_path or any(not part or part.isdigit() for part in parts):
            raise _CliConfigError(f"invalid --set path {raw_path!r}; sequence indexes are unsupported")
        if command == "predict" and parts[0] in {"optimization", "optimizer"}:
            raise _CliConfigError(f"--set path {raw_path!r} is not in the schema")
        current: Any = data
        for part in parts[:-1]:
            if not isinstance(current, dict):
                raise _CliConfigError(f"--set path {raw_path!r} crosses a non-mapping value")
            if part not in current:
                current[part] = {}
            current = current[part]
        leaf = parts[-1]
        if not isinstance(current, dict):
            raise _CliConfigError(f"--set path {raw_path!r} crosses a non-mapping value")
        try:
            current[leaf] = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise _CliConfigError(f"invalid YAML value for --set {raw_path!r}: {exc}") from exc


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
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise _CliExecutionError(f"{type(exc).__name__}: {exc}") from exc
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
        native["summary"]["metric_semantics"] = report.metadata["metric_semantics"]
        native["summary"]["total_gpus"] = report.metadata["total_gpus"]
    summary = prediction_summary(native)
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
    sys.stdout.write(format_prediction_stdout(summary, args.format, details=details))
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
    if not selected:
        sys.stderr.write(f"no feasible candidate found; saved full result to: {result_path}\n")
        return 1
    paths = write_recommendations(root, [config for _, _, config in selected])
    rows = [
        {
            "rank": index,
            "score": candidate.score,
            "objectives": candidate.objectives,
            "used_gpus": candidate.used_gpus,
            "config_path": str(path),
        }
        for index, ((_, candidate, _), path) in enumerate(zip(selected, paths, strict=True), start=1)
    ]
    sys.stdout.write(format_recommendation_stdout(rows, args.format))
    sys.stdout.write("\n")
    if args.format == "table":
        sys.stdout.write(f"Saved full result to: {result_path}\n")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    from aiconfigurator.cli.main import _resolve_cli_log_level, run_estimate
    from aiconfigurator.logging_utils import setup_logging
    from aiconfigurator.sdk import perf_database

    setup_logging(level=_resolve_cli_log_level(args), no_color=args.no_color)
    try:
        perf_database.set_systems_paths(args.systems_paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_estimate(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(estimate_arguments=arguments[:1] == ["estimate"])
    args = parser.parse_args(arguments)
    if args.command == "estimate":
        try:
            return _estimate(args)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            sys.stderr.write(f"aisimulate estimate failed: {type(exc).__name__}: {exc}\n")
            return 1
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
    except _CliExecutionError as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
