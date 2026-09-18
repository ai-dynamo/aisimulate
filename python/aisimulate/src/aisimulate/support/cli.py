# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal setup and local planning for FPM onboarding."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import ValidationError

from aisimulate.config.common import load_yaml

from .config_profile import (
    FIELD_CHOICES,
    INTEGER_FIELDS,
    OVERRIDE_FIELDS,
    ModelConfig,
    ProfileDraft,
    ProfileRequestError,
    derive_profile,
    load_model_config,
    validate_overrides,
)
from .fpm import run_fpm
from .plan import create_plan, request_id
from .schema import FPMDeployment, SearchProfile, SloSpec, SupportIdentity, SupportRequest, WorkloadSpec


def add_support_parser(subparsers: Any) -> None:
    onboard = subparsers.add_parser(
        "onboard",
        help="Onboard a model for FPM simulation on a target hardware platform.",
        description=(
            "Onboard a model for FPM simulation on a target hardware platform. "
            "Declare the model and hardware, generate a plan, and preview or execute FPM collection."
        ),
    )
    actions = onboard.add_subparsers(dest="support_action", required=True)
    init = actions.add_parser(
        "init",
        help="Declare a model, target hardware, and FPM pilot workload.",
        description=(
            "Use --interactive for terminal prompts. --model-config derives supported local model metadata; "
            "supply unresolved profile fields with --resource-overrides. Otherwise scripted setup requires "
            "--model, --model-revision, --model-kind, --framework-version, --gpu, --gpu-count, and --interconnect. "
            "One node and a small TP1 synthetic pilot are defaults; review the saved scope before collection."
        ),
    )
    init.add_argument(
        "--interactive",
        action="store_true",
        help="Guide setup with terminal prompts; review and accept config-based profiles before saving.",
    )
    init.add_argument("--profile", choices=("onboarding",), help="Optional alias; all setup uses onboarding.")
    init.add_argument("--model")
    init.add_argument("--model-revision", help="Pinned model revision; moving labels such as main are unsupported.")
    init.add_argument("--model-kind", choices=("dense", "moe"))
    init.add_argument("--framework", choices=("vllm",), help="Collection runtime (default: vllm).")
    init.add_argument("--framework-version", help="Pinned vLLM version in the collection environment.")
    init.add_argument("--gpu", help="Target GPU system name, for example h200_sxm.")
    init.add_argument("--gpu-count", type=int, help="Total GPUs available.")
    init.add_argument("--node-count", type=int, help="Number of nodes (default: 1).")
    init.add_argument("--gpus-per-node", type=int, help="GPUs available per node; defaults to gpu-count on one node.")
    init.add_argument("--interconnect", help="Interconnect, for example nvswitch, pcie, or none.")
    init.add_argument("--sm", type=int, help="GPU SM version override.")
    init.add_argument("--tokenizer-revision")
    init.add_argument("--chat-template-revision")
    init.add_argument("--aisimulate-revision", help="Optional pinned AISimulate source revision.")
    init.add_argument("--tensor-parallel", type=int, help="Attention tensor-parallel size (default: 1).")
    init.add_argument("--attention-data-parallel", type=int, help="Attention data-parallel size (default: 1).")
    init.add_argument("--moe-tensor-parallel", type=int, help="Expert tensor-parallel size (default: TP for MoE).")
    init.add_argument("--moe-expert-parallel", type=int, help="Expert-parallel size (default: 1).")
    profile_source = init.add_mutually_exclusive_group()
    profile_source.add_argument(
        "--fpm-profile", help="JSON/YAML identity and resource profile for class-independent FPM."
    )
    profile_source.add_argument("--model-config", metavar="PATH", help="Local Hugging Face-style config.json.")
    init.add_argument(
        "--resource-overrides",
        metavar="PATH",
        help="Flat JSON/YAML profile field overrides; requires --model-config. Resource bytes are per rank.",
    )
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

    plan = actions.add_parser(
        "plan", help="Plan FPM collection and write predict/recommend configs for the target hardware."
    )
    plan.add_argument("-c", "--config", required=True)
    plan.add_argument("--output-dir", default="./aisimulate-support")
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--format", choices=("table", "json"), default="table")

    collect = actions.add_parser(
        "collect-fpm", help="Preview FPM collection on the target hardware, or run with --execute."
    )
    collect.add_argument("-c", "--config", required=True)
    collect.add_argument("--output-dir", default="./aisimulate-support")
    collect.add_argument(
        "--execute", action="store_true", help="Execute collection using a matching saved onboarding plan."
    )
    collect.add_argument(
        "--smoke", action="store_true", help="Diagnostic smoke collection; does not publish formal data."
    )
    collect.add_argument(
        "--limit", type=int, help="Limit diagnostic cases; requires --smoke, no formal data publication."
    )
    collect.add_argument("--resume", action="store_true", help="Resume the existing collector checkpoint.")
    collect.add_argument("--checkpoint-dir", help="Checkpoint directory inside the plan's fpm-checkpoint directory.")
    deployment = collect.add_argument_group("Collector deployment")
    deployment.add_argument("--dynamo-version", help="Target Dynamo release used to resolve collector templates.")
    deployment.add_argument("--image", help="Collector container image; prefer an immutable digest.")
    deployment.add_argument("--namespace", help="Kubernetes namespace for collector resources.")
    deployment.add_argument("--model-cache", metavar="NAME[:MOUNT[:SUBPATH]]", help="Model-cache PVC and mount.")
    deployment.add_argument("--transport", choices=("nvlink", "ib", "efa"))
    deployment.add_argument("--image-pull-secret", help="Kubernetes secret for pulling the collector image.")


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
            **({"fpm_profile": load_yaml(args.fpm_profile)} if getattr(args, "fpm_profile", None) else {}),
        }
    )


def _request_target(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path).expanduser().absolute()
    if target.exists() and not target.is_file():
        raise ValueError(f"onboarding request {target} must be a file path")
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise ValueError(f"onboarding request {target} exists; pass --overwrite to replace it")
    parent = target.parent
    while not parent.exists() and not parent.is_symlink():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError(f"onboarding request parent {parent} must be a writable directory")
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
    "gpu": ("Target GPU system name (for example h200_sxm)", str),
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
    "attention_data_parallel": ("Attention data-parallel size", int),
    "moe_tensor_parallel": ("Expert tensor-parallel size", int),
    "moe_expert_parallel": ("Expert-parallel size", int),
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


def _require_terminal() -> None:
    if not sys.stdin.isatty():
        raise ValueError(
            "--interactive requires a terminal; for automation supply options (aisimulate onboard init --help)"
        )


def _guided_request(args: argparse.Namespace) -> SupportRequest:
    _require_terminal()
    print("Onboard a model for FPM simulation on a target hardware platform.")
    print("Start with a pure-TP worker and a small synthetic workload on vLLM.")
    print("Enter accepts a displayed default. Ctrl-C cancels without saving. Supplied options skip their prompts.")
    for name in _PROMPTS:
        if getattr(args, name) is None:
            _prompt(args, name)
    return _validate_guided_request(args)


def _correct_option(args: argparse.Namespace, exc: ValueError, *, require_known_field: bool = False) -> None:
    name, message = None, str(exc)
    if isinstance(exc, ProfileRequestError):
        name = exc.field
    if isinstance(exc, ValidationError):
        error = exc.errors(include_url=False)[0]
        name = error["loc"][-1] if error["loc"] else None
        message = error["msg"]
    if name == "backend_version":
        name = "framework_version"
    if require_known_field and name not in _CORRECTION_PROMPTS:
        raise exc
    print(f"{name or 'Request'}: {message}")
    while name not in _CORRECTION_PROMPTS:
        name = input("Option to correct (for example --tensor-parallel): ").strip().removeprefix("--").replace("-", "_")
        if name not in _CORRECTION_PROMPTS:
            print("Choose one of: " + ", ".join(key.replace("_", "-") for key in _CORRECTION_PROMPTS))
    _prompt(args, name)


def _validate_guided_request(args: argparse.Namespace) -> SupportRequest:
    while True:
        try:
            return _request_from_args(args)
        except ValidationError as exc:
            _correct_option(args, exc)


class _ResourceOverridesLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        values = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("resource override field names must be strings")
            if key in values:
                raise ValueError(f"duplicate resource override field: {key}")
            values[key] = self.construct_object(value_node, deep=deep)
        return values


def _load_resource_overrides(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path).expanduser()
    try:
        values = yaml.load(source.read_text(encoding="utf-8"), Loader=_ResourceOverridesLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed resource overrides in {source}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"resource overrides in {source} must be a flat JSON/YAML mapping")
    return values


def _memory_field(name: str) -> bool:
    return name.endswith("_bytes") or name == "kv_bytes_per_token"


def _resource_answer(name: str, answer: str) -> str | int:
    if name not in INTEGER_FIELDS:
        return answer
    if _memory_field(name):
        units = {"B": 1, **{f"{prefix}B": 1000**power for power, prefix in enumerate("KMGT", 1)}}
        units.update({f"{prefix}IB": 1024**power for power, prefix in enumerate("KMGT", 1)})
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([KMGT]i?B|B)", answer, re.IGNORECASE)
        if match:
            value = Fraction(match[1]) * units[match[2].upper()]
            if value.denominator != 1:
                raise ValueError("memory quantities must resolve to a whole number of bytes")
            return value.numerator
        try:
            return int(answer)
        except ValueError as exc:
            raise ValueError("enter integer bytes or an explicit unit, for example 70 GiB or 512 MiB") from exc
    try:
        return int(answer)
    except ValueError as exc:
        raise ValueError("enter a valid integer") from exc


def _read_profile_value(name: str) -> dict[str, Any]:
    choices = f" ({', '.join(FIELD_CHOICES[name])})" if name in FIELD_CHOICES else ""
    if _memory_field(name):
        choices += " (per rank; integer bytes or units such as GiB/MiB)"
    while True:
        answer = input(f"{name}{choices}: ").strip()
        if not answer:
            print("A value is required.")
            continue
        try:
            return validate_overrides({name: _resource_answer(name, answer)})
        except ValueError as exc:
            print(f"{name}: {exc}")


def _prompt_profile_value(
    config: ModelConfig,
    request: SupportRequest | None,
    overrides: dict[str, Any],
    draft: ProfileDraft,
    name: str,
) -> tuple[dict[str, Any], ProfileDraft]:
    print(f"{name}: {draft.missing[name]}")
    # Retain a valid answer if final profile validation asks for an identity
    # correction. Other derivation failures cannot be repaired by repeating
    # this answer, so let the caller report the actual error.
    overrides.update(_read_profile_value(name))
    updated = derive_profile(config, request, overrides)
    print(f"  {name}: {updated.resolved[name]} (source: {updated.sources[name]})")
    return overrides, updated


def _review_config_profile(
    config: ModelConfig,
    request: SupportRequest,
    overrides: dict[str, Any],
    draft: ProfileDraft,
) -> SupportRequest:
    while True:
        identity, workload = request.identity, request.workload
        print("Review FPM profile before saving:")
        print(f"  Modeling scope: {config.notes['modeling_scope']}")
        print(
            f"  Deployment: {identity.model} ({identity.model_kind}) @ {identity.model_revision}; "
            f"{identity.framework} {identity.framework_version}; "
            f"{identity.gpu_count} {identity.gpu} GPU(s) on {identity.node_count} node(s); {identity.interconnect}"
        )
        print(f"  Parallelism: {request.parallelism()}")
        print(
            f"  Pilot: {workload.input_tokens}/{workload.output_tokens} tokens, concurrency {workload.concurrency}, "
            f"{workload.request_count} requests; context limit {request.search.context_length} tokens"
        )
        print("  Resource *_bytes values are bytes per rank; kv_bytes_per_token is bytes per cached token per rank.")
        print("  max_num_tokens/max_batch_size are per-rank scheduler limits. Estimates require runtime verification.")
        for name, value in draft.resolved.items():
            print(f"  {name}: {value} (source: {draft.sources[name]})")
        if "provenance" not in draft.resolved:
            print(
                "  Provenance records the config SHA-256, deployment, values and sources; "
                "edit provenance to add a note."
            )
        print("Nothing has been saved. Accept saves these values; edit changes a profile field. Ctrl-C cancels setup.")
        while True:
            action = input("Review action (accept/edit/cancel): ").strip().lower()
            if action in ("accept", "edit", "cancel"):
                break
            print("Enter accept, edit, or cancel. Explicit acceptance is required to save.")
        if action == "accept":
            return request
        if action == "cancel":
            raise KeyboardInterrupt
        print("Editable fields: " + ", ".join(sorted(OVERRIDE_FIELDS)))
        while True:
            name = input("Field to edit: ").strip()
            if name in OVERRIDE_FIELDS:
                break
            print("Choose an editable field by its displayed name.")
        # Derive from explicit inputs again so inferred dependents can change.
        # Only publish the staged values after the entire request validates.
        staged = {**overrides, **_read_profile_value(name)}
        try:
            updated = derive_profile(config, request, staged)
            while updated.missing:
                field = min(updated.missing, key=_memory_field)
                staged, updated = _prompt_profile_value(config, request, staged, updated, field)
            if updated.profile is None:
                raise ValueError("model config did not produce a complete FPM profile")
            validated = SupportRequest.model_validate({**request.model_dump(), "fpm_profile": updated.profile})
        except ValueError as exc:
            print(f"Edit rejected: {exc}. Previous profile values retained.")
            continue
        overrides, draft, request = staged, updated, validated


def _config_request(args: argparse.Namespace) -> SupportRequest:
    config = load_model_config(args.model_config)
    overrides = _load_resource_overrides(args.resource_overrides)
    preview = derive_profile(config, None, overrides)
    print(f"Model config source: {Path(args.model_config).expanduser()} (SHA-256 {config.sha256}).")
    for name, note in config.notes.items():
        print(f"Config {name}: {note}")
    for name in ("architecture", "context_length", "num_experts"):
        if name in preview.resolved:
            print(f"{name}: {preview.resolved[name]} (source: {preview.sources[name]})")
    if args.interactive:
        _require_terminal()
        for name in ("architecture", "context_length", "num_experts"):
            if name in preview.missing:
                overrides, preview = _prompt_profile_value(config, None, overrides, preview, name)
    if args.model is None:
        args.model = config.suggestions.get("model")
    if args.model_kind is None:
        experts = preview.resolved.get("num_experts")
        args.model_kind = (
            ("moe" if experts > 0 else "dense") if experts is not None else config.suggestions.get("model_kind")
        )
    if args.context_length is None and "context_length" in preview.resolved:
        args.context_length = min(
            preview.resolved["context_length"], SearchProfile.model_fields["context_length"].default
        )
    if args.interactive:
        request = _guided_request(args)
    else:
        try:
            request = _request_from_args(args)
        except ValidationError as exc:
            details = []
            for error in exc.errors(include_url=False):
                name = error["loc"][-1] if error["loc"] else None
                option = "--" + name.replace("_", "-") if name in _CORRECTION_PROMPTS else "request"
                details.append(f"  {option}: {error['msg']}")
            raise ValueError(
                "Complete deployment identity and pilot options before dependent profile derivation:\n"
                + "\n".join(details)
                + "\nThen supply unresolved profile fields with --resource-overrides PATH (flat JSON/YAML), "
                "or use --interactive."
            ) from exc
    while True:
        try:
            draft = derive_profile(config, request, overrides)
        except (ProfileRequestError, ValidationError) as exc:
            if not args.interactive:
                raise
            _correct_option(args, exc, require_known_field=True)
            request = _validate_guided_request(args)
            continue
        print("Profile inputs (resource bytes per rank; estimates require runtime verification):")
        for name, value in draft.resolved.items():
            print(f"  {name}: {value} (source: {draft.sources[name]})")
        if draft.missing and not args.interactive:
            raise ValueError(
                "Unresolved profile fields for this deployment:\n"
                + "\n".join(f"  {name}: {reason}" for name, reason in draft.missing.items())
                + "\nSupply these flat fields in --resource-overrides PATH (JSON/YAML; integer bytes per rank), "
                "or use --interactive."
            )
        try:
            while draft.missing:
                # Resolve dtype/layout/envelope inputs before asking for dependent memory amounts.
                name = min(draft.missing, key=_memory_field)
                overrides, draft = _prompt_profile_value(config, request, overrides, draft, name)
        except (ProfileRequestError, ValidationError) as exc:
            _correct_option(args, exc, require_known_field=True)
            request = _validate_guided_request(args)
            continue
        if draft.profile is None:
            raise ValueError("model config did not produce a complete FPM profile")
        try:
            completed = SupportRequest.model_validate({**request.model_dump(), "fpm_profile": draft.profile})
        except ValidationError as exc:
            if not args.interactive:
                raise
            _correct_option(args, exc)
            request = _validate_guided_request(args)
            continue
        return _review_config_profile(config, completed, overrides, draft) if args.interactive else completed


def _init(args: argparse.Namespace) -> int:
    if args.resource_overrides and not args.model_config:
        raise ValueError("--resource-overrides requires --model-config")
    _request_target(args.output, overwrite=args.overwrite)
    try:
        if args.model_config:
            request = _config_request(args)
        else:
            request = _guided_request(args) if args.interactive else _request_from_args(args)
        path = _write_request(request, args.output, overwrite=args.overwrite)
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled.", file=sys.stderr)
        return 130
    workload = request.workload
    print(
        f"Scope: FPM simulation for {request.identity.model} ({request.identity.model_kind}), "
        f"vLLM {request.identity.framework_version}, "
        f"{request.identity.gpu_count} {request.identity.gpu} GPU(s) on {request.identity.node_count} node(s); "
        f"{request.parallel_preset} {request.parallelism()}, {workload.input_tokens}/{workload.output_tokens} tokens, "
        f"concurrency {workload.concurrency}, {workload.request_count} requests, "
        f"up to {request.search.max_candidates} candidate(s)."
    )
    if args.model_config:
        print(
            "Request saved with the complete FPM profile and declared or estimated resources. "
            "Runtime compatibility and FPM data are unchecked; accuracy is not assessed. "
            "Setup has not launched GPU work."
        )
    else:
        print(
            "Request saved. Model metadata/resources, runtime compatibility and FPM data are unchecked; "
            "accuracy is not assessed. Setup has not launched GPU work."
        )
    _print(
        {
            "request_id": request_id(request),
            "request": str(path),
            "next": shlex.join(["aisimulate", "onboard", "plan", "--config", str(path)]),
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
            "prerequisites": "runtime_and_fpm_data_unchecked" if request.fpm_profile is not None else "unchecked",
            **({"resources": "estimated_from_declared_profile"} if request.fpm_profile is not None else {}),
            "accuracy": "not assessed",
            "next": shlex.join(
                [
                    "aisimulate",
                    "onboard",
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
            deployment=FPMDeployment(**_values(args, FPMDeployment)),
        )
    raise AssertionError(f"unhandled onboarding action {args.support_action!r}")
