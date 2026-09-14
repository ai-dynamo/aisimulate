# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal setup and local planning for FPM onboarding."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import ValidationError

from .fpm import run_fpm
from .plan import create_plan, request_id
from .schema import SearchProfile, SloSpec, SupportIdentity, SupportRequest, WorkloadSpec


def add_support_parser(subparsers: Any) -> None:
    support = subparsers.add_parser(
        "support",
        help="Set up a model and GPU allocation for local FPM collection.",
        description="Create an onboarding request, generate a plan, and preview or execute local FPM collection.",
    )
    actions = support.add_subparsers(dest="support_action", required=True)
    init = actions.add_parser(
        "init",
        help="Create an onboarding request with prompts or scripted options.",
        description=(
            "Use --interactive for terminal prompts. Scripted setup requires --model, --model-revision, "
            "--model-kind, --framework-version, --gpu, --gpu-count, and --interconnect. "
            "One node and a small TP1 synthetic pilot are defaults; review the saved scope before collection."
        ),
    )
    init.add_argument("--interactive", action="store_true", help="Guide setup with terminal prompts.")
    init.add_argument("--profile", choices=("onboarding",), help="Optional alias; all setup uses onboarding.")
    init.add_argument("--model")
    init.add_argument("--model-revision", help="Pinned model revision; moving labels such as main are unsupported.")
    init.add_argument("--model-kind", choices=("dense", "moe"))
    init.add_argument("--framework", choices=("vllm",), help="Collection runtime (default: vllm).")
    init.add_argument("--framework-version", help="Pinned vLLM version in the collection environment.")
    init.add_argument("--gpu", help="GPU system name, for example h200_sxm.")
    init.add_argument("--gpu-count", type=int, help="Total GPUs available.")
    init.add_argument("--node-count", type=int, help="Number of nodes (default: 1).")
    init.add_argument("--gpus-per-node", type=int, help="GPUs available per node; defaults to gpu-count on one node.")
    init.add_argument("--interconnect", help="Interconnect, for example nvswitch, pcie, or none.")
    init.add_argument("--sm", type=int, help="GPU SM version override.")
    init.add_argument("--tokenizer-revision")
    init.add_argument("--chat-template-revision")
    init.add_argument("--aisimulate-revision", help="Optional pinned AISimulate source revision.")
    init.add_argument("--tensor-parallel", type=int, help="GPUs per pure-TP worker (default: 1).")
    init.add_argument("--input-tokens", type=int, help="Input tokens per request (default: 1024).")
    init.add_argument("--output-tokens", type=int, help="Output tokens per request (default: 128).")
    init.add_argument("--concurrency", type=int, help="Concurrent requests (default: 1).")
    init.add_argument("--request-count", type=int, help="Synthetic request count (default: 4).")
    init.add_argument("--context-length", type=int, help="Pilot context limit in tokens (default: 16384).")
    init.add_argument("--ttft-ms", type=float, help="Target time to first token in ms (default: 1000).")
    init.add_argument("--tpot-ms", type=float, help="Target time per output token in ms (default: 100).")
    init.add_argument("--max-candidates", type=int, help="One worker, or also the largest replica count (1 or 2).")
    init.add_argument(
        "--objective",
        choices=get_args(SearchProfile.model_fields["objective"].annotation),
        help="Recommendation objective (default: throughput).",
    )
    init.add_argument("--seed", type=int, help="Recommendation search seed (default: 42).")
    init.add_argument("--output", default="support-request.yaml")
    init.add_argument("--overwrite", action="store_true")

    plan = actions.add_parser("plan", help="Write ordinary predict/recommend configs and local collector commands.")
    plan.add_argument("-c", "--config", required=True)
    plan.add_argument("--output-dir", default="./aisimulate-support")
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--format", choices=("table", "json"), default="table")

    collect = actions.add_parser("collect-fpm", help="Preview the local collector command, or run with --execute.")
    collect.add_argument("-c", "--config", required=True)
    collect.add_argument("--output-dir", default="./aisimulate-support")
    collect.add_argument(
        "--execute", action="store_true", help="Execute collection using a matching saved support plan."
    )
    collect.add_argument(
        "--smoke", action="store_true", help="Diagnostic smoke collection; does not publish formal data."
    )
    collect.add_argument(
        "--limit", type=int, help="Limit diagnostic cases; requires --smoke, no formal data publication."
    )
    collect.add_argument("--resume", action="store_true", help="Resume the existing collector checkpoint.")
    collect.add_argument("--checkpoint-dir", help="Checkpoint directory inside the plan's fpm-checkpoint directory.")


def _values(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    return {name: getattr(args, name) for name in model.model_fields if getattr(args, name, None) is not None}


def _request_from_args(args: argparse.Namespace) -> SupportRequest:
    identity = _values(args, SupportIdentity)
    if args.gpus_per_node is None and identity.get("node_count", 1) == 1:
        identity["gpus_per_node"] = args.gpu_count
    return SupportRequest.model_validate(
        {
            "identity": identity,
            "workload": {**_values(args, WorkloadSpec), "slo": _values(args, SloSpec)},
            "search": _values(args, SearchProfile),
        }
    )


def _request_target(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path).expanduser().absolute()
    if target.exists() and not target.is_file():
        raise ValueError(f"support request {target} must be a file path")
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise ValueError(f"support request {target} exists; pass --overwrite to replace it")
    parent = target.parent
    while not parent.exists() and not parent.is_symlink():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError(f"support request parent {parent} must be a writable directory")
    return target


def _write_request(request: SupportRequest, path: str | Path, *, overwrite: bool) -> Path:
    target = _request_target(path, overwrite=overwrite)
    payload = yaml.safe_dump(request.model_dump(mode="json", exclude_none=True), sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _print(value: dict[str, Any], output_format: str = "table") -> None:
    if output_format == "json":
        print(json.dumps(value, sort_keys=True))
        return
    for key, item in value.items():
        rendered = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        print(f"{key}: {rendered}")


_PROMPTS = {
    "model": ("Model name or path", str),
    "model_revision": ("Pinned model revision (not main/latest)", str),
    "model_kind": ("Model kind (dense/moe)", str),
    "framework_version": ("Pinned vLLM version", str),
    "gpu": ("GPU system name (for example h200_sxm)", str),
    "gpu_count": ("Total GPUs available", int),
    "interconnect": ("GPU interconnect (for example nvswitch, pcie, or none)", str),
    "tensor_parallel": ("GPUs per pure-TP worker", int),
    "input_tokens": ("Input tokens per request", int),
    "output_tokens": ("Output tokens per request", int),
    "concurrency": ("Concurrent requests", int),
    "context_length": ("Pilot context limit in tokens", int),
    "ttft_ms": ("Target time to first token (ms)", float),
    "tpot_ms": ("Target time per output token (ms)", float),
}
_CORRECTION_PROMPTS = {
    **_PROMPTS,
    "framework": ("Runtime (vllm)", str),
    "tokenizer_revision": ("Pinned tokenizer revision", str),
    "chat_template_revision": ("Pinned chat-template revision", str),
    "aisimulate_revision": ("Pinned AISimulate revision", str),
    "node_count": ("Number of nodes", int),
    "gpus_per_node": ("GPUs per node", int),
    "sm": ("GPU SM version", int),
    "request_count": ("Number of synthetic requests", int),
    "max_candidates": ("Maximum candidates (1 or 2)", int),
    "objective": ("Recommendation objective", str),
    "seed": ("Recommendation search seed", int),
}


def _prompt(args: argparse.Namespace, name: str) -> None:
    label, convert = _CORRECTION_PROMPTS[name]
    default = getattr(args, name)
    if default is None:
        for model in (SupportIdentity, WorkloadSpec, SloSpec, SearchProfile):
            field = model.model_fields.get(name)
            if field is not None and not field.is_required():
                default = field.default
        if name == "gpu_count":
            default = 1
    while True:
        answer = input(f"{label}{f' [{default}]' if default is not None else ''}: ").strip()
        if not answer and default is None:
            print("A value is required.")
            continue
        try:
            value = convert(answer) if answer else default
        except ValueError:
            print(f"Enter a valid {'integer' if convert is int else 'number'}.")
            continue
        setattr(args, name, value)
        return


def _guided_request(args: argparse.Namespace) -> SupportRequest:
    if not sys.stdin.isatty():
        raise ValueError(
            "--interactive requires a terminal; for automation supply options (aisimulate support init --help)"
        )
    print("FPM onboarding: a pure-TP worker and a small synthetic workload on vLLM.")
    print("Enter accepts a displayed default. Ctrl-C cancels without saving. Supplied options skip their prompts.")
    for name in _PROMPTS:
        if getattr(args, name) is None:
            _prompt(args, name)
    while True:
        try:
            return _request_from_args(args)
        except ValidationError as exc:
            error = exc.errors(include_url=False)[0]
            name = error["loc"][-1] if error["loc"] else None
            print(f"{name or 'Request'}: {error['msg']}")
        while name not in _CORRECTION_PROMPTS:
            name = (
                input("Option to correct (for example --tensor-parallel): ")
                .strip()
                .removeprefix("--")
                .replace("-", "_")
            )
            if name not in _CORRECTION_PROMPTS:
                print("Choose one of: " + ", ".join(key.replace("_", "-") for key in _CORRECTION_PROMPTS))
        _prompt(args, name)


def _init(args: argparse.Namespace) -> int:
    _request_target(args.output, overwrite=args.overwrite)
    try:
        request = _guided_request(args) if args.interactive else _request_from_args(args)
        path = _write_request(request, args.output, overwrite=args.overwrite)
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled.", file=sys.stderr)
        return 130
    workload = request.workload
    print(
        f"Scope: {request.identity.model_kind} model, vLLM {request.identity.framework_version}, "
        f"{request.identity.gpu_count} {request.identity.gpu} GPU(s) on {request.identity.node_count} node(s); "
        f"TP{request.search.tensor_parallel}, {workload.input_tokens}/{workload.output_tokens} tokens, "
        f"concurrency {workload.concurrency}, {workload.request_count} requests, "
        f"up to {request.search.max_candidates} candidate(s)."
    )
    print(
        "Request saved. Model integration, runtime compatibility and FPM data are unchecked; "
        "accuracy is not assessed. Setup has not launched GPU work."
    )
    _print(
        {
            "request_id": request_id(request),
            "request": str(path),
            "next": shlex.join(["aisimulate", "support", "plan", "--config", str(path)]),
        }
    )
    return 0


def _plan(args: argparse.Namespace) -> int:
    request = SupportRequest.from_yaml(args.config)
    root = Path(args.output_dir).expanduser().resolve()
    plan = create_plan(request, root, overwrite=args.overwrite)
    _print(
        {
            "request_id": plan["request_id"],
            "candidate_count": plan["search"]["candidate_count"],
            "plan": str(root / "support-plan.json"),
            "commands": plan["outputs"]["commands"],
            "prerequisites": "unchecked",
            "accuracy": "not assessed",
            "next": shlex.join(
                [
                    "aisimulate",
                    "support",
                    "collect-fpm",
                    "--config",
                    plan["outputs"]["request"],
                    "--output-dir",
                    str(root),
                ]
            ),
        },
        args.format,
    )
    return 0


def run_support_command(args: argparse.Namespace) -> int:
    if args.support_action == "init":
        return _init(args)
    if args.support_action == "plan":
        return _plan(args)
    if args.support_action == "collect-fpm":
        return run_fpm(
            SupportRequest.from_yaml(args.config),
            output_dir=args.output_dir,
            execute=args.execute,
            smoke=args.smoke,
            limit=args.limit,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
        )
    raise AssertionError(f"unhandled support action {args.support_action!r}")
