# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line surface for the exact-cell support workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from aisimulate import __version__
from aisimulate.output import prepare_output_directory

from .fpm import run_fpm
from .identity import support_cell_id
from .plan import create_plan, existing_support
from .schema import (
    ExecutionProfile,
    FPMProfile,
    SearchProfile,
    SloSpec,
    SupportIdentity,
    SupportRequest,
    ValidationPolicy,
    WorkloadSpec,
)
from .validation import load_evidence, validate_evidence


def add_support_parser(subparsers: Any) -> None:
    support = subparsers.add_parser(
        "support",
        help="Create and validate an exact model/GPU support cell.",
        description=(
            "Run the self-service workflow when no published support cell exists: "
            "init, check/plan, collect FPM evidence, and validate."
        ),
    )
    actions = support.add_subparsers(dest="support_action", required=True)

    init = actions.add_parser("init", help="Create an exact MVP support request.")
    init.add_argument("--model", required=True)
    init.add_argument("--model-revision", required=True)
    init.add_argument("--tokenizer-revision", default=None)
    init.add_argument("--chat-template-revision", default=None)
    init.add_argument("--model-kind", choices=("auto", "dense", "moe"), default="auto")
    init.add_argument("--framework", choices=("vllm", "sglang", "trtllm"), default="vllm")
    init.add_argument("--framework-version", required=True)
    init.add_argument("--gpu", required=True)
    init.add_argument("--gpu-count", type=int, required=True)
    init.add_argument("--node-count", type=int, default=1)
    init.add_argument("--gpus-per-node", type=int, default=None)
    init.add_argument("--interconnect", required=True)
    init.add_argument("--sm", type=int, default=None)
    init.add_argument("--agentx-trace", required=True)
    init.add_argument("--agentx-digest", required=True)
    init.add_argument(
        "--trace-format",
        choices=("mooncake", "mooncake-delta", "agentic_mooncake", "applied_compute_agentic", "dynamo"),
        default="dynamo",
    )
    init.add_argument("--concurrency", type=int, default=10)
    init.add_argument("--request-count", type=int, default=100)
    init.add_argument("--ttft-ms", type=float, required=True)
    init.add_argument("--tpot-ms", type=float, required=True)
    init.add_argument("--max-candidates", type=int, default=16)
    init.add_argument("--context-length", type=int, default=16384)
    init.add_argument(
        "--objective",
        choices=(
            "throughput",
            "throughput_per_gpu",
            "throughput_per_user",
            "goodput",
            "goodput_per_gpu",
            "ttft",
            "e2e_latency",
            "pareto",
        ),
        default="throughput",
    )
    init.add_argument("--recommendation-uplift-min", type=float, default=None)
    init.add_argument("--brev-instance", default=None)
    init.add_argument("--aisimulate-revision", default=__version__)
    init.add_argument("--output", default="support-request.yaml")
    init.add_argument("--overwrite", action="store_true")

    action_help = {
        "check": "Check the packaged exact-cell support matrix.",
        "plan": "Create the bounded search, FPM, Brev, and evidence artifacts.",
        "collect-fpm": "Print or execute the frozen whole-model FPM campaign.",
        "validate": "Apply fail-closed database, FPM, E2E, and value gates.",
    }
    for action, help_text in action_help.items():
        child = actions.add_parser(action, help=help_text, description=help_text)
        child.add_argument("-c", "--config", required=True)
    actions.choices["check"].add_argument("--format", choices=("table", "json"), default="table")
    actions.choices["plan"].add_argument("--output-dir", default="./aisimulate-support")
    actions.choices["plan"].add_argument("--overwrite", action="store_true")
    actions.choices["plan"].add_argument("--format", choices=("table", "json"), default="table")

    collect = actions.choices["collect-fpm"]
    collect.add_argument("--execute", action="store_true", help="Run instead of printing the frozen FPM plan.")
    collect.add_argument("--smoke", action="store_true")
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--checkpoint-dir", default=None)
    collect.add_argument("--output-dir", default="./aisimulate-support")

    validate = actions.choices["validate"]
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--systems-root", required=True)
    validate.add_argument("--output-dir", default="./aisimulate-support-validation")
    validate.add_argument("--overwrite", action="store_true")
    validate.add_argument("--format", choices=("table", "json"), default="table")


def _load_request(path: str | Path) -> SupportRequest:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read support request {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed support request YAML {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"support request {source} must contain one mapping")
    return SupportRequest.model_validate(raw)


def _request_from_args(args: argparse.Namespace) -> SupportRequest:
    if args.gpu_count < 1 or args.node_count < 1:
        raise ValueError("gpu-count and node-count must be positive")
    if args.gpus_per_node is None:
        if args.gpu_count % args.node_count:
            raise ValueError("gpu-count must be divisible by node-count when gpus-per-node is omitted")
        gpus_per_node = args.gpu_count // args.node_count
    else:
        gpus_per_node = args.gpus_per_node
    slo = SloSpec(ttft_ms=args.ttft_ms, tpot_ms=args.tpot_ms)
    identity = SupportIdentity(
        model=args.model,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision or args.model_revision,
        chat_template_revision=args.chat_template_revision or args.model_revision,
        model_kind=args.model_kind,
        framework=args.framework,
        framework_version=args.framework_version,
        gpu=args.gpu,
        gpu_count=args.gpu_count,
        node_count=args.node_count,
        gpus_per_node=gpus_per_node,
        interconnect=args.interconnect,
        sm=args.sm,
        aisimulate_revision=args.aisimulate_revision,
    )
    workloads = [
        WorkloadSpec(
            id="fixed-8k-1k",
            kind="synthetic",
            input_tokens=8192,
            output_tokens=1024,
            concurrency=args.concurrency,
            request_count=args.request_count,
            slo=slo,
        ),
        WorkloadSpec(
            id="agentx",
            kind="trace",
            trace_path=args.agentx_trace,
            trace_digest=args.agentx_digest,
            trace_format=args.trace_format,
            concurrency=args.concurrency,
            request_count=args.request_count,
            slo=slo,
        ),
    ]
    return SupportRequest(
        identity=identity,
        workloads=workloads,
        search=SearchProfile(
            max_candidates=args.max_candidates,
            objective=args.objective,
            context_length=args.context_length,
        ),
        fpm=FPMProfile(),
        validation=ValidationPolicy(recommendation_uplift_min=args.recommendation_uplift_min),
        execution=ExecutionProfile(instance=args.brev_instance),
    )


def _write_request(request: SupportRequest, path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise ValueError(f"support request {target} exists; pass --overwrite to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(request.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    return target


def _print(value: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            rendered = json.dumps(item, sort_keys=True)
        else:
            rendered = str(item)
        sys.stdout.write(f"{key}: {rendered}\n")


def _init(args: argparse.Namespace) -> int:
    request = _request_from_args(args)
    path = _write_request(request, args.output, overwrite=args.overwrite)
    _print(
        {
            "support_cell_id": support_cell_id(request),
            "request": str(path),
            "next": f"aisimulate support plan --config {path}",
        },
        "table",
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    request = _load_request(args.config)
    status = existing_support(request)
    value = {
        "support_cell_id": support_cell_id(request),
        **status.model_dump(mode="json", exclude_none=True),
        "next": (
            "use aisimulate predict/recommend with the published exact cell"
            if status.status == "supported"
            else f"aisimulate support plan --config {args.config}"
        ),
    }
    _print(value, args.format)
    return 0 if status.status == "supported" else 1


def _plan(args: argparse.Namespace) -> int:
    request = _load_request(args.config)
    plan = create_plan(request, args.output_dir, overwrite=args.overwrite)
    _print(
        {
            "support_cell_id": plan["support_cell_id"],
            "existing_support": plan["existing_support"]["status"],
            "candidate_count": plan["search"]["candidate_count"],
            "fpm_status": plan["fpm"]["status"],
            "plan": str(Path(args.output_dir) / "support-plan.json"),
            "next": str(Path(args.output_dir) / "commands.json"),
        },
        args.format,
    )
    return 0


def _collect_fpm(args: argparse.Namespace) -> int:
    request = _load_request(args.config)
    return run_fpm(
        request,
        execute=args.execute,
        smoke=args.smoke,
        limit=args.limit,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )


def _validate(args: argparse.Namespace) -> int:
    request = _load_request(args.config)
    evidence = load_evidence(args.evidence)
    result = validate_evidence(request, evidence, systems_root=args.systems_root)
    root = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    result_path = root / "validation.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print(
        {
            "support_cell_id": result.support_cell_id,
            "status": result.status,
            "validation": str(result_path),
            "errors": result.errors,
        },
        args.format,
    )
    return 0 if result.status == "pass" else 1


def run_support_command(args: argparse.Namespace) -> int:
    if args.support_action == "init":
        return _init(args)
    if args.support_action == "check":
        return _check(args)
    if args.support_action == "plan":
        return _plan(args)
    if args.support_action == "collect-fpm":
        return _collect_fpm(args)
    if args.support_action == "validate":
        return _validate(args)
    raise AssertionError(f"unhandled support action {args.support_action!r}")
