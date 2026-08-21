# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single public AISimulate command-line application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .compiler import prediction_to_replay_spec
from .config_adapter import (
    ConfigAdapterResolutionError,
    resolve_config_adapters,
)
from .output import (
    format_prediction_stdout,
    format_recommendation_stdout,
    prepare_output_directory,
    write_prediction_report,
    write_recommendations,
    write_requests,
)
from .public_config import PredictionConfig, RecommendationConfig, known_config_path
from .stack import StackResolutionError, resolve_runner_factory
from .sweeper.replay import ReplayOutputRequirements


class _CliConfigError(ValueError):
    pass


class _CliExecutionError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisimulate",
        description="Predict or recommend an offline LLM serving configuration.",
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
    subparsers.choices["predict"].add_argument(
        "--capture-per-request", action="store_true"
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


def _apply_overrides(
    data: dict[str, Any], overrides: list[str], *, command: str
) -> None:
    for assignment in overrides:
        if "=" not in assignment:
            raise _CliConfigError(
                f"invalid --set {assignment!r}; expected PATH=YAML_VALUE"
            )
        raw_path, raw_value = assignment.split("=", 1)
        parts = raw_path.split(".")
        if not raw_path or any(not part or part.isdigit() for part in parts):
            raise _CliConfigError(
                f"invalid --set path {raw_path!r}; sequence indexes are unsupported"
            )
        if not known_config_path(raw_path, command=command):
            raise _CliConfigError(f"--set path {raw_path!r} is not in the schema")
        current: Any = data
        for part in parts[:-1]:
            if not isinstance(current, dict):
                raise _CliConfigError(
                    f"--set path {raw_path!r} crosses a non-mapping value"
                )
            if part not in current:
                current[part] = {}
            current = current[part]
        leaf = parts[-1]
        if not isinstance(current, dict):
            raise _CliConfigError(
                f"--set path {raw_path!r} crosses a non-mapping value"
            )
        try:
            current[leaf] = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise _CliConfigError(
                f"invalid YAML value for --set {raw_path!r}: {exc}"
            ) from exc


def _adapter_names_for_prediction(config: PredictionConfig, stack: str) -> list[str]:
    names = []
    if config.router.policy != "round_robin":
        names.append(f"{stack}.router")
    if config.planner.policy != "disabled":
        names.append(f"{stack}.planner")
    return names


def _adapter_names_for_recommendation(
    config: RecommendationConfig, stack: str
) -> list[str]:
    names = []
    if config.router is not None:
        policy = config.router.get("policy", "round_robin")
        if policy != "round_robin" or isinstance(policy, dict):
            names.append(f"{stack}.router")
    if config.planner is not None:
        policy = config.planner.get("policy", "disabled")
        if policy != "disabled" or isinstance(policy, dict) or any(
            name in config.planner
            for name in (
                "scaling_policy",
                "fpm_sampling",
                "load_sensitivity",
                "load_predictor",
            )
        ):
            names.append(f"{stack}.planner")
    return names


def _predict(args: argparse.Namespace, raw: dict[str, Any], factory) -> int:
    config = PredictionConfig.model_validate(raw)
    names = _adapter_names_for_prediction(config, args.stack)
    adapters = resolve_config_adapters(names)
    spec = prediction_to_replay_spec(config, stack=args.stack, adapters=adapters)
    factory.capabilities().require_compatible(spec)
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    runner = factory.create(0)
    try:
        try:
            report = runner.run(
                spec,
                output_requirements=ReplayOutputRequirements(
                    include_raw_report=True,
                    capture_per_request=args.capture_per_request,
                ),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise _CliExecutionError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        runner.close()
    native = report.metadata.get("native_report")
    if not isinstance(native, dict):
        native = {"summary": dict(report.metrics)}
    summary = native.get("summary", native)
    if not isinstance(summary, dict):
        raise RuntimeError("prediction report summary must be a JSON mapping")
    report_path = write_prediction_report(root, native)
    if args.capture_per_request:
        records = native.get("per_request")
        if not isinstance(records, list):
            raise RuntimeError(
                "selected stack did not provide per-request prediction records"
            )
        if any(not isinstance(record, dict) for record in records):
            raise RuntimeError("per-request records must be JSON mappings")
        write_requests(root, records)
    sys.stdout.write(format_prediction_stdout(summary, args.format))
    sys.stdout.write("\n")
    if args.format == "table":
        sys.stdout.write(f"Saved full report to: {report_path}\n")
    return 0


def _recommend(args: argparse.Namespace, raw: dict[str, Any], factory) -> int:
    from .recommend import run_recommendation

    config = RecommendationConfig.model_validate(raw)
    names = _adapter_names_for_recommendation(config, args.stack)
    adapters = resolve_config_adapters(names)
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    candidates = run_recommendation(
        config,
        stack=args.stack,
        runner_factory=factory,
        providers=adapters,
        show_progress=args.format == "table",
    )
    if not candidates:
        sys.stderr.write("no feasible candidate found\n")
        return 1
    concrete: list[PredictionConfig] = []
    for candidate in candidates:
        if candidate.prediction_config is None:
            raise RuntimeError("recommendation candidate has no concrete public config")
        concrete.append(PredictionConfig.model_validate(candidate.prediction_config))
    paths = write_recommendations(root, concrete)
    rows = [
        {
            "rank": index,
            "score": candidate.score,
            "objectives": candidate.objectives,
            "used_gpus": candidate.used_gpus,
            "config_path": str(path),
        }
        for index, (candidate, path) in enumerate(
            zip(candidates, paths, strict=True), start=1
        )
    ]
    sys.stdout.write(format_recommendation_stdout(rows, args.format))
    sys.stdout.write("\n")
    return 0


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
    except _CliExecutionError as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"aisimulate {args.command} failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
