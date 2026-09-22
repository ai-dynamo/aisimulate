# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated command line for whole-model FPM campaigns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from collector.model_cases import build_collection_case_plan

from .config import add_fpm_arguments, add_fpm_generator_arguments
from .entry import _load_generator_overrides, resolve_inputs, resolve_run_inputs, run_resolved

_INPUT_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m collector.fpm_forward",
        description="Plan or run a Generator-resolved Dynamo-native FPM campaign.",
    )
    parser.add_argument("--backend", choices=("vllm",), default="vllm")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-architecture", default=None)
    parser.add_argument("--model-cases", default=None, help="Optional model cases YAML path.")
    parser.add_argument("--gpu", default=None, help="Target AIC system, required for a new campaign.")
    parser.add_argument("--sm", type=int, default=None, help="Optional explicit SM version for case planning.")
    parser.add_argument("--plan-only", action="store_true", help="Print the frozen FPM plan and exit.")
    parser.add_argument("--smoke", action="store_true", help="Run the minimal smoke sampling profile.")
    parser.add_argument("--limit", type=int, default=None, help="Limit cells; allowed only with --smoke.")
    parser.add_argument("--resume", action="store_true", help="Resume the matching frozen-plan checkpoint.")
    parser.add_argument(
        "--resume-retry-failed",
        action="store_true",
        help="Retry failed cells while resuming; requires --resume.",
    )
    parser.add_argument("--checkpoint-dir", default=".collector_checkpoint")
    validation = parser.add_argument_group("Validation subset collection")
    validation.add_argument("--repeatability-source-campaign", default=None, help="Existing frozen campaign directory.")
    validation.add_argument(
        "--repeatability-source-checkpoint", default=None, help="Its passed fpm_forward.json checkpoint."
    )
    validation.add_argument(
        "--repeatability-output-dir", default=None, help="Fresh validation output; original data is preserved."
    )
    validation.add_argument("--repeatability-samples", type=int, default=5)
    validation.add_argument("--repeatability-max-points", type=int, default=12)
    validation.add_argument("--repeatability-cv-threshold", type=float, default=0.05)
    add_fpm_arguments(parser)
    add_fpm_generator_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.model_path or args.model_architecture or args.model_cases or args.repeatability_source_campaign):
        parser.error("FPM requires --model-path, --model-architecture, or --model-cases")
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")
    repeatability_paths = (
        args.repeatability_source_campaign,
        args.repeatability_source_checkpoint,
        args.repeatability_output_dir,
    )
    repeatability = any(repeatability_paths)
    if repeatability and not all(repeatability_paths):
        parser.error("repeatability requires source campaign, source checkpoint and a separate output directory")
    if repeatability and (args.smoke or args.limit is not None):
        parser.error("repeatability cannot use --smoke or --limit")
    if not repeatability and not args.gpu:
        parser.error("FPM requires --gpu for a new campaign")

    try:
        if repeatability:
            from .repeatability import (
                freeze_repeatability_plan,
                load_repeatability_deployment,
                load_repeatability_source,
                run_repeatability,
            )

            plan = load_repeatability_source(args.repeatability_source_campaign)
            generator_overrides = _load_generator_overrides(args)
            if not generator_overrides and not args.plan_only:
                generator_overrides = load_repeatability_deployment(args.repeatability_source_campaign)
            kwargs = {
                "samples": args.repeatability_samples,
                "max_points_per_cell": args.repeatability_max_points,
                "cv_threshold": args.repeatability_cv_threshold,
            }
            if args.plan_only:
                result = freeze_repeatability_plan(
                    plan, args.repeatability_source_campaign, args.repeatability_source_checkpoint, **kwargs
                )
            else:
                result = run_repeatability(
                    plan,
                    generator_overrides=generator_overrides,
                    source_campaign_dir=args.repeatability_source_campaign,
                    source_checkpoint_path=args.repeatability_source_checkpoint,
                    output_dir=args.repeatability_output_dir,
                    resume=args.resume,
                    retry_failed=args.resume_retry_failed,
                    **kwargs,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if args.plan_only or result["status"] == "passed" else 1
        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
        )
        if case_plan.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = case_plan.model_path
        if args.plan_only:
            plan, _generator_overrides = resolve_inputs(args, case_plan)
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        resolved_inputs = resolve_run_inputs(args, case_plan)
    except _INPUT_ERRORS as error:
        parser.error(str(error))

    errors = run_resolved(args, resolved_inputs)
    if errors:
        print(json.dumps(errors, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
