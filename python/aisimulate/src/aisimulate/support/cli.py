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
from copy import deepcopy
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
from .schema import (
    AGENTX_REFERENCE_CONTEXT,
    CollectionSpec,
    FPMDeployment,
    SearchProfile,
    SloSpec,
    SupportIdentity,
    SupportRequest,
    WorkloadSpec,
)
from .topology import TopologySuggestions, suggest_topologies


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
        help="Declare a model, target hardware, worker topology, and FPM collection limits.",
        description=(
            "Use --interactive for terminal prompts. --model-config derives supported local model metadata; "
            "supply unresolved profile fields with --resource-overrides. Otherwise scripted setup requires "
            "--model, --model-revision, --model-kind, --framework-version, --gpu, and --interconnect. "
            "Collection GPUs are derived from the selected topology. Model-config setup suggests model/hardware-aware "
            "topologies; other routes default to TP1. AISimulate launches benchmark workers when collection is "
            "executed; no separately launched HTTP server is required. An existing vLLM launch configuration is "
            "optional evidence for reviewing settings, not a required input or automatically imported file. "
            "AISimulate sets runtime limits and optional capture overrides; Dynamo self-benchmark combines them "
            "with image sampling defaults and runtime "
            "feasibility checks to generate the exact grid. A complete grid does not establish query coverage. "
            "Synthetic workload and SLA options customize optional validation examples only. "
            "Set deployment replicas and GPU budgets in ordinary predict/recommend configs."
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
    init.add_argument("--interconnect", help="Interconnect, for example nvswitch, pcie, or none.")
    init.add_argument("--sm", type=int, help="GPU SM version override.")
    init.add_argument("--tokenizer-revision")
    init.add_argument("--chat-template-revision")
    init.add_argument("--aisimulate-revision", help="Optional pinned AISimulate source revision.")
    init.add_argument(
        "--tensor-parallel", type=int, help="Attention tensor-parallel size; bypasses model-config suggestions."
    )
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
        help="JSON/YAML profile field overrides, including cache_groups/cache_block_sizes; "
        "requires --model-config. Resource bytes are per rank.",
    )
    init.add_argument(
        "--suggest-parallel",
        action="store_true",
        help="Preview model-config topology suggestions as JSON; no prompts or writes, including to --output.",
    )
    init.add_argument(
        "--parallel-configs",
        metavar="PATH",
        help="JSON/YAML list of explicit parallel configurations with optional per-entry resource_overrides; "
        "requires --model-config and --output-dir; conflicts with topology flags and --suggest-parallel.",
    )
    init.add_argument("--input-tokens", type=int, help="Synthetic validation input tokens (default: 1024).")
    init.add_argument("--output-tokens", type=int, help="Synthetic validation output tokens (default: 128).")
    init.add_argument("--concurrency", type=int, help="Synthetic validation concurrency (default: 1).")
    init.add_argument("--request-count", type=int, help="Synthetic validation request count (default: 4).")
    init.add_argument(
        "--context-length",
        type=int,
        help="Runtime per-request context limit; defaults to min(model/profile context, AgentX reference 256000).",
    )
    init.add_argument(
        "--max-num-tokens", type=int, help="Rank-local scheduled token budget; profile bound or initial policy 8192."
    )
    init.add_argument(
        "--max-batch-size",
        type=int,
        help="Rank-local scheduler sequence bound; profile bound or initial policy 256. "
        "Does not request every prefill batch.",
    )
    init.add_argument(
        "--prefill-cudagraph-policy",
        choices=("runtime", "explicit"),
        help="Prefill CUDA graphs: runtime selection (new setup default), or explicit capture override.",
    )
    init.add_argument(
        "--max-prefill-cudagraph-size",
        type=int,
        help="Select explicit capture with this limit; explicit policy without a limit uses 2048. "
        "Conflicts with runtime policy; match the serving target.",
    )
    init.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Fraction of total GPU memory for collection and simulation, in (0, 1] (initial policy: 0.90).",
    )
    init.add_argument("--ttft-ms", type=float, help="Synthetic validation TTFT target in ms (default: 1000).")
    init.add_argument("--tpot-ms", type=float, help="Synthetic validation TPOT target in ms (default: 100).")
    init.add_argument(
        "--objective",
        choices=get_args(SearchProfile.model_fields["objective"].annotation),
        help="Recommendation objective (default: throughput).",
    )
    init.add_argument("--seed", type=int, help="Recommendation search seed (default: 42).")
    output = init.add_mutually_exclusive_group()
    output.add_argument(
        "--output", default="support-request.yaml", help="Save one request (default: support-request.yaml)."
    )
    output.add_argument(
        "--output-dir",
        metavar="ROOT",
        help="Save a request and FPM profile for each selected configuration in a fresh directory; "
        "requires --model-config. Interactive suggestions accept comma-separated choices.",
    )
    init.add_argument(
        "--overwrite", action="store_true", help="Allow replacement of --output; unsupported with --output-dir."
    )

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
    from .validation import add_validation_parser

    add_validation_parser(actions)


def _values(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    return {name: getattr(args, name) for name in model.model_fields if getattr(args, name, None) is not None}


def _request_from_args(args: argparse.Namespace) -> SupportRequest:
    collection = _values(args, CollectionSpec)
    collection.setdefault(
        "prefill_cudagraph_policy", "explicit" if "max_prefill_cudagraph_size" in collection else "runtime"
    )
    collection.setdefault("gpu_memory_utilization", 0.9)
    return SupportRequest.model_validate(
        {
            "identity": _values(args, SupportIdentity),
            "workload": {**_values(args, WorkloadSpec), "slo": _values(args, SloSpec)},
            "search": _values(args, SearchProfile),
            "collection": collection,
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
    request.scheduler_limits()
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
    "interconnect": ("GPU interconnect (for example nvswitch, pcie, or none)", str),
    "tensor_parallel": ("Attention tensor-parallel size", int),
}
_CORRECTION_PROMPTS = {
    **_PROMPTS,
    "input_tokens": ("Input tokens per request", int),
    "output_tokens": ("Output tokens per request", int),
    "concurrency": ("Concurrent requests", int),
    "context_length": ("Runtime per-request context limit in tokens", int),
    "ttft_ms": ("Target time to first token (ms)", float),
    "tpot_ms": ("Target time per output token (ms)", float),
    "max_num_tokens": ("Rank-local scheduled token budget", int),
    "max_batch_size": ("Rank-local scheduler sequence bound", int),
    "max_prefill_cudagraph_size": ("Prefill CUDA graph capture limit", int),
    "prefill_cudagraph_policy": ("Prefill CUDA graph policy (runtime/explicit)", str),
    "gpu_memory_utilization": ("Fraction of total GPU memory (0 < value <= 1)", float),
    "framework": ("Runtime (vllm)", str),
    "tokenizer_revision": ("Pinned tokenizer revision", str),
    "chat_template_revision": ("Pinned chat-template revision", str),
    "aisimulate_revision": ("Pinned AISimulate revision", str),
    "sm": ("GPU SM version", int),
    "request_count": ("Number of synthetic requests", int),
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
        for model in (SupportIdentity, WorkloadSpec, SloSpec, SearchProfile, CollectionSpec):
            field = model.model_fields.get(name)
            if field is not None and not field.is_required():
                default = field.default
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
        if name == "prefill_cudagraph_policy" and value == "runtime":
            args.max_prefill_cudagraph_size = None
        elif name == "max_prefill_cudagraph_size":
            args.prefill_cudagraph_policy = "explicit"
        return


def _require_terminal() -> None:
    if not sys.stdin.isatty():
        raise ValueError(
            "--interactive requires a terminal; for automation supply options (aisimulate onboard init --help)"
        )


def _guided_request(args: argparse.Namespace, *, skip_topology: bool = False) -> SupportRequest:
    _require_terminal()
    print("Onboard a model for FPM simulation on a target hardware platform.")
    selection = "one or more" if getattr(args, "output_dir", None) is not None else "one"
    print(f"Select {selection} TP, DEP, or TEP worker configurations and review collection limits for vLLM.")
    print("AISimulate launches benchmark workers during collection; you do not need to start an HTTP server.")
    print("An existing vLLM launch configuration is optional evidence for reviewing these settings.")
    print("Enter accepts a displayed default. Ctrl-C cancels without saving. Supplied options skip their prompts.")
    for name in _PROMPTS:
        if skip_topology and name == "tensor_parallel":
            continue
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


def _resource_answer(name: str, answer: str) -> Any:
    if name in {"cache_groups", "cache_block_sizes"}:
        try:
            return yaml.load(answer, Loader=_ResourceOverridesLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"enter a valid JSON {'list' if name == 'cache_groups' else 'object'}") from exc
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
    elif name == "cache_groups":
        choices += " (JSON list; page_size_bytes includes every group layer and runtime padding per rank)"
    elif name == "cache_block_sizes":
        choices += " (JSON object mapping displayed group names to runtime tokens per block)"
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
    answer = _read_profile_value(name)
    if name == "cache_block_sizes":
        answer[name] = {**overrides.get(name, {}), **answer[name]}
    overrides.update(answer)
    updated = derive_profile(config, request, overrides)
    if name in updated.resolved:
        print(f"  {name}: {updated.resolved[name]} (source: {updated.sources[name]})")
    else:
        print(f"  {name}: {updated.missing[name]}")
    return overrides, updated


def _review_config_profile(
    config: ModelConfig,
    request: SupportRequest,
    overrides: dict[str, Any],
    draft: ProfileDraft,
    *,
    directory_output: bool = False,
) -> SupportRequest:
    while True:
        identity = request.identity
        print("Review FPM profile before saving:")
        print(f"  Modeling scope: {config.notes['modeling_scope']}")
        print(
            f"  Target: {identity.model} ({identity.model_kind}) @ {identity.model_revision}; "
            f"{identity.framework} {identity.framework_version}; "
            f"{identity.gpu}; {identity.interconnect}"
        )
        print(f"  Parallelism: {request.parallelism()}")
        print(f"  Collection GPUs required: {request.worker_gpus} for one selected worker; availability is unchecked.")
        settings = request.collection_settings()
        print(f"  Runtime context limit: {request.search.context_length} tokens per request.")
        print(f"  Collection: {json.dumps(settings, sort_keys=True)}")
        print("  AISimulate launches and manages benchmark workers when collection is executed.")
        print("  vLLM loads the model and allocates caches; Dynamo self-benchmark generates and measures the grid.")
        print("  AISimulate supplies runtime limits and collection policies; no HTTP server is needed beforehand.")
        print(
            "  Dynamo combines them with image sampling defaults and runtime feasibility checks to generate the grid."
        )
        if request.collection.prefill_cudagraph_policy == "runtime":
            print("  CUDA graph sizes resolve in the initialized runtime; no explicit capture override is proposed.")
        else:
            print("  Prefill capture overrides configure the engine; match the serving target.")
        print("  The sequence bound does not request every prefill batch; total KV capacity resolves at runtime.")
        print("  A complete generated grid does not establish AgentX/direct-FPM query coverage.")
        print("  Synthetic validation traffic and SLAs do not determine these collection limits.")
        print("  Resource *_bytes values are bytes per rank; kv_bytes_per_token is bytes per cached token per rank.")
        if draft.resolved.get("cache_layout") == "grouped":
            print("  Grouped pages retain full history or the declared window, with separate convolution state.")
            print("  Group page_size_bytes sums every group layer per rank; verify runtime block sizes and padding.")
            print(
                "  Config-derived pages are minimum packed tensor estimates; edit cache_groups for runtime allocation."
            )
            print("  Logical context determines FPM timing queries; retained cache length is only memory accounting.")
        print("  max_num_tokens/max_batch_size are per-rank scheduler limits. Estimates require runtime verification.")
        for name, value in draft.resolved.items():
            print(f"  {name}: {value} (source: {draft.sources[name]})")
        if draft.resolved.get("comm_quant_mode") != "half":
            print(
                "  Collection compatibility: this collector currently supports comm_quant_mode=half. "
                "Your override is retained, but new collection will reject a mismatched identity."
            )
        if "provenance" not in draft.resolved:
            print(
                "  Provenance records the config SHA-256, deployment, values and sources; "
                "edit provenance to add a note."
            )
        if directory_output:
            print("Nothing has been saved. All selected profiles must be accepted before any files are saved.")
            print("Accept keeps this profile; edit changes a profile field. Ctrl-C cancels setup.")
        else:
            print(
                "Nothing has been saved. Accept saves these values; edit changes a profile field. Ctrl-C cancels setup."
            )
        while True:
            action = input("Review action (accept/edit/cancel): ").strip().lower()
            if action in ("accept", "edit", "cancel"):
                break
            print("Enter accept, edit, or cancel. Explicit acceptance is required to save.")
        if action == "accept":
            return request
        if action == "cancel":
            raise KeyboardInterrupt
        collection_fields = {
            "runtime_context_length",
            "max_prefill_cudagraph_size",
            "prefill_cudagraph_policy",
            "gpu_memory_utilization",
        }
        print("Editable fields: " + ", ".join(sorted(OVERRIDE_FIELDS | collection_fields)))
        while True:
            name = input("Field to edit: ").strip()
            if name in OVERRIDE_FIELDS | collection_fields:
                break
            print("Choose an editable field by its displayed name.")
        # Derive from explicit inputs again so inferred dependents can change.
        # Only publish the staged values after the entire request validates.
        staged = deepcopy(overrides)
        payload = request.model_dump(exclude={"fpm_profile"})
        if name in collection_fields:
            if name == "prefill_cudagraph_policy":
                while (value := input(f"{name} (runtime/explicit): ").strip()) not in {"runtime", "explicit"}:
                    print("Enter runtime or explicit.")
                if value == "runtime":
                    payload["collection"]["max_prefill_cudagraph_size"] = None
            elif name == "gpu_memory_utilization":
                while True:
                    try:
                        value = float(input(f"{name} (0 < fraction <= 1): ").strip())
                        if not 0 < value <= 1:
                            raise ValueError
                        break
                    except ValueError:
                        print("Enter a finite fraction greater than 0 and at most 1.")
            else:
                while True:
                    try:
                        value = int(input(f"{name} (positive integer): ").strip())
                        if value <= 0:
                            raise ValueError
                        break
                    except ValueError:
                        print("Enter a positive integer.")
            if name == "runtime_context_length":
                payload["search"]["context_length"] = value
            else:
                payload["collection"][name] = value
                if name == "max_prefill_cudagraph_size":
                    payload["collection"]["prefill_cudagraph_policy"] = "explicit"
        else:
            staged.update(_read_profile_value(name))
            if name in {"max_num_tokens", "max_batch_size"}:
                payload["collection"][name] = staged[name]
        try:
            edited_request = SupportRequest.model_validate(payload)
            updated = derive_profile(config, edited_request, staged)
            while updated.missing:
                field = min(updated.missing, key=_memory_field)
                staged, updated = _prompt_profile_value(config, edited_request, staged, updated, field)
            if updated.profile is None:
                raise ValueError("model config did not produce a complete FPM profile")
            validated = SupportRequest.model_validate({**edited_request.model_dump(), "fpm_profile": updated.profile})
        except ValueError as exc:
            print(f"Edit rejected: {exc}. Previous profile values retained.")
            continue
        overrides, draft, request = staged, updated, validated


_TOPOLOGY_OPTIONS = ("tensor_parallel", "attention_data_parallel", "moe_tensor_parallel", "moe_expert_parallel")


def _explicit_topology(args: argparse.Namespace) -> bool:
    return any(getattr(args, name) is not None for name in _TOPOLOGY_OPTIONS)


def _config_inputs(args: argparse.Namespace) -> tuple[ModelConfig, dict[str, Any], SupportRequest]:
    config = load_model_config(args.model_config)
    overrides = _load_resource_overrides(args.resource_overrides)
    preview = derive_profile(config, None, overrides)
    if not args.suggest_parallel:
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
        args.context_length = min(preview.resolved["context_length"], AGENTX_REFERENCE_CONTEXT)
    if args.interactive:
        request = _guided_request(args, skip_topology=not _explicit_topology(args))
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
                "Complete target identity and collection options before dependent profile derivation:\n"
                + "\n".join(details)
                + "\nThen supply unresolved profile fields with --resource-overrides PATH (flat JSON/YAML), "
                "or use --interactive."
            ) from exc
    return config, overrides, request


def _topology_summary(suggestions: TopologySuggestions) -> str:
    lines = [
        "Suggested worker topologies (collection starting points, not a performance ranking):",
        f"  Hardware: {suggestions.hardware.source}; {suggestions.hardware.assumption}",
    ]
    for number, candidate in enumerate(suggestions.candidates, 1):
        default = " (default)" if candidate == suggestions.default else ""
        lines.append(
            f"  {number}. {candidate.family}, {candidate.required_gpus} GPUs, {candidate.status}{default}: "
            + shlex.join(candidate.cli_flags)
        )
        lines.extend(f"     {reason}" for reason in candidate.reasons)
        lines.extend(f"     Missing {name}: {reason}" for name, reason in candidate.missing.items())
    if not suggestions.candidates:
        lines.append("  No eligible topology remains in the automatic hardware domain.")
        for candidate in suggestions.rejected_candidates:
            lines.append(
                f"  Rejected {candidate.family}, {candidate.required_gpus} GPUs: " + "; ".join(candidate.reasons)
            )
    lines.append("Runtime compatibility, actual GPU availability and measured performance remain unchecked.")
    return "\n".join(lines)


def _resolve_shared_profile_inputs(
    config: ModelConfig, shared: dict[str, Any], entry_overrides: list[dict[str, Any]]
) -> None:
    while True:
        for overrides in entry_overrides:
            draft = derive_profile(config, None, {**shared, **overrides})
            pending = [
                name
                for name in draft.missing
                if not _memory_field(name) and name not in {"cache_groups", "cache_block_sizes"}
            ]
            if pending:
                name = pending[0]
                print(f"{name}: {draft.missing[name]}")
                shared.update(_read_profile_value(name))
                break
        else:
            return


def _select_topologies(
    args: argparse.Namespace,
    config: ModelConfig,
    overrides: dict[str, Any],
    request: SupportRequest,
    *,
    multiple: bool = False,
) -> list[SupportRequest]:
    while True:
        try:
            suggestions = suggest_topologies(config, request, overrides)
            if args.interactive and suggestions.candidates:
                _resolve_shared_profile_inputs(config, overrides, [{}])
                suggestions = suggest_topologies(config, request, overrides)
            break
        except (ProfileRequestError, ValidationError) as exc:
            if not args.interactive:
                raise
            _correct_option(args, exc, require_known_field=True)
            request = _validate_guided_request(args)
    summary = _topology_summary(suggestions)
    if not suggestions.candidates or (not args.interactive and suggestions.default is None):
        raise ValueError(
            "No fully assessed topology default is available.\n"
            + summary
            + "\nSupply shared precision/layout inputs with --resource-overrides or use --interactive. "
            "For rank-local byte bounds, select an exact topology with the displayed flags first. "
            "Explicit topology may be outside the automatic shortlist."
        )
    print(summary)
    selected = [suggestions.default] if suggestions.default is not None else []
    if args.interactive:
        default = suggestions.candidates.index(selected[0]) + 1 if selected else None
        if default is None:
            print("Memory fit is unresolved; select a candidate explicitly, then provide its missing per-rank bounds.")
        while True:
            answer = input(
                f"Choose {'topologies' if multiple else 'topology'} 1-{len(suggestions.candidates)}"
                + (f" [{default}]" if default is not None else "")
                + (" (comma-separated, or cancel): " if multiple else " (or cancel): ")
            ).strip()
            if answer.lower() == "cancel":
                raise KeyboardInterrupt
            if not answer and default is not None:
                break
            choices = [value.strip() for value in answer.split(",")] if multiple else [answer]
            if all(value.isdecimal() and 1 <= int(value) <= len(suggestions.candidates) for value in choices) and len(
                {int(value) for value in choices}
            ) == len(choices):
                selected = [suggestions.candidates[int(value) - 1] for value in choices]
                break
            numbers = "unique comma-separated candidate numbers" if multiple else "a displayed candidate number"
            print(f"Enter {numbers} or cancel. Enter accepts only a fully assessed default.")
    for candidate in selected:
        print(
            f"Selected {candidate.family} with {candidate.required_gpus} collection GPUs: "
            + shlex.join(candidate.cli_flags)
        )
    return [candidate.apply(request) for candidate in selected]


def _topology_values(request: SupportRequest) -> dict[str, int]:
    parallel = request.parallelism()
    return dict(
        zip(
            _TOPOLOGY_OPTIONS,
            (parallel[name] for name in ("tensor", "attention_data", "moe_tensor", "moe_expert")),
            strict=True,
        )
    )


def _config_request(args: argparse.Namespace) -> SupportRequest:
    automatic = not _explicit_topology(args)
    config, overrides, request = _config_inputs(args)
    if automatic:
        request = _select_topologies(args, config, overrides, request)[0]
        for name, value in _topology_values(request).items():
            setattr(args, name, value)
    return _complete_config_request(args, config, overrides, request)


def _complete_config_request(
    args: argparse.Namespace, config: ModelConfig, overrides: dict[str, Any], request: SupportRequest
) -> SupportRequest:
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
                + "\nInspect the pinned runtime and checkpoint metadata for unresolved precision/layout facts. "
                "An existing launch configuration is optional evidence; values not established locally remain "
                "explicit inputs. Supply them in --resource-overrides PATH (JSON/YAML; bytes per rank), "
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
        return (
            _review_config_profile(
                config, completed, overrides, draft, directory_output=getattr(args, "output_dir", None) is not None
            )
            if args.interactive
            else completed
        )


def _load_parallel_configs(path: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    try:
        entries = yaml.load(source.read_text(encoding="utf-8"), Loader=_ResourceOverridesLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed parallel configurations in {source}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("--parallel-configs must contain a nonempty JSON/YAML list of objects")
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"parallel configuration {number} must be an object")
        unknown = entry.keys() - {*_TOPOLOGY_OPTIONS, "resource_overrides"}
        if unknown:
            raise ValueError(f"parallel configuration {number} has unknown fields: {', '.join(sorted(unknown))}")
        if "tensor_parallel" not in entry:
            raise ValueError(f"parallel configuration {number} requires tensor_parallel")
        for name in entry.keys() & set(_TOPOLOGY_OPTIONS):
            if type(entry[name]) is not int or entry[name] < 1:
                raise ValueError(f"parallel configuration {number}: {name} must be a positive integer")
        overrides = entry.get("resource_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"parallel configuration {number}: resource_overrides must be a mapping")
        entry["resource_overrides"] = validate_overrides(overrides)
    return entries


def _check_unique_topologies(requests: list[SupportRequest]) -> None:
    seen = set()
    for request in requests:
        topology = _topology_values(request)
        key = tuple(topology.values())
        if key in seen:
            raise ValueError(f"duplicate resolved parallel configuration: {topology}")
        seen.add(key)


def _config_requests(args: argparse.Namespace) -> list[SupportRequest]:
    entries = _load_parallel_configs(args.parallel_configs) if args.parallel_configs is not None else None
    automatic = not _explicit_topology(args) and entries is None
    implicit_context = args.context_length is None
    config, shared, request = _config_inputs(args)
    if entries is not None:
        selected = []
        for entry in entries:
            payload = request.model_dump()
            payload["search"].update({name: entry[name] for name in _TOPOLOGY_OPTIONS if name in entry})
            selected.append(SupportRequest.model_validate(payload))
    else:
        selected = _select_topologies(args, config, shared, request, multiple=True) if automatic else [request]
    _check_unique_topologies(selected)
    if len(selected) > 1 and (
        rank_fields := [name for name in shared if _memory_field(name) or name == "cache_groups"]
    ):
        raise ValueError(
            "Multiple configurations cannot share rank-local resource overrides: "
            + ", ".join(sorted(rank_fields))
            + ". Put them in each --parallel-configs entry's resource_overrides or edit each profile interactively."
        )
    if args.interactive and not automatic:
        _resolve_shared_profile_inputs(
            config, shared, [entry["resource_overrides"] for entry in entries] if entries is not None else [{}]
        )
    completed = []
    for number, candidate in enumerate(selected, 1):
        # Corrections, nested resource inputs and review edits belong only to
        # this tuple. Start every profile from the original shared intake.
        candidate_args = deepcopy(args)
        candidate = candidate.model_copy(deep=True)
        for name, value in _topology_values(candidate).items():
            setattr(candidate_args, name, value)
        overrides = deepcopy(shared)
        if entries is not None:
            overrides.update(deepcopy(entries[number - 1]["resource_overrides"]))
        if implicit_context:
            context = derive_profile(config, None, overrides).resolved.get("context_length")
            if context is not None:
                candidate_args.context_length = min(context, AGENTX_REFERENCE_CONTEXT)
                candidate.search.context_length = candidate_args.context_length
        print(f"Configuration {number}/{len(selected)}: {_topology_values(candidate)}")
        completed.append(_complete_config_request(candidate_args, config, overrides, candidate))
        _check_unique_topologies(completed)
    return completed


def _onboarding_target(path: str | Path) -> Path:
    if not str(path):
        raise ValueError("onboarding output directory must be a nonempty path")
    target = Path(path).expanduser().absolute()
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ValueError(f"onboarding output path must not contain symlinks: {parent}")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(
            f"onboarding output directory {target} must be new or empty; collection results cannot be replaced"
        )
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError(f"onboarding output parent {parent} must be a writable directory")
    return target


def _write_onboarding(requests: list[SupportRequest], root: Path) -> dict[str, Any]:
    _check_unique_topologies(requests)
    for request in requests:
        request.scheduler_limits()
    root = _onboarding_target(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": "aisimulate-onboarding/v1", "configurations": []}
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        staged = Path(temporary)
        for request in requests:
            topology = _topology_values(request)
            tp, dp, mtp, ep = topology.values()
            name = f"tp{tp}-dp{dp}-moe-tp{mtp}-moe-ep{ep}"
            final = root / name
            _write_request(request, staged / name / "request.yaml", overwrite=False)
            profile = request.fpm_profile
            assert profile is not None
            (staged / name / "fpm-profile.json").write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
            index["configurations"].append(
                {
                    "topology": topology,
                    "collection_gpus_required": request.worker_gpus,
                    "request_id": request_id(request),
                    "request": str(final / "request.yaml"),
                    "fpm_profile": str(final / "fpm-profile.json"),
                    "collection_dir": str(final / "collection"),
                    "plan_command": shlex.join(
                        [
                            "aisimulate",
                            "onboard",
                            "plan",
                            "--config",
                            str(final / "request.yaml"),
                            "--output-dir",
                            str(final / "collection"),
                        ]
                    ),
                }
            )
        (staged / "onboarding.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        # Recheck after all reviews and serialization. Rename publishes only a
        # complete directory and cannot replace a nonempty collection directory.
        _onboarding_target(root)
        staged.rename(root)
    return index


def _init(args: argparse.Namespace) -> int:
    if args.resource_overrides and not args.model_config:
        raise ValueError("--resource-overrides requires --model-config")
    if args.parallel_configs is not None:
        if not args.model_config or args.output_dir is None:
            raise ValueError("--parallel-configs requires --model-config and --output-dir")
        if _explicit_topology(args):
            raise ValueError("--parallel-configs cannot be combined with explicit topology flags")
    if args.output_dir is not None:
        if not args.model_config:
            raise ValueError("--output-dir requires --model-config")
        if args.overwrite:
            raise ValueError("--output-dir cannot be combined with --overwrite; use a fresh output directory")
    if args.suggest_parallel:
        if not args.model_config:
            raise ValueError("--suggest-parallel requires --model-config")
        if (
            args.interactive
            or args.fpm_profile
            or args.profile
            or _explicit_topology(args)
            or args.output_dir is not None
            or args.parallel_configs is not None
        ):
            raise ValueError(
                "--suggest-parallel cannot be combined with --interactive, profiles, explicit topology flags, "
                "--output-dir, or --parallel-configs"
            )
        config, overrides, request = _config_inputs(args)
        _print(suggest_topologies(config, request, overrides).to_dict(), "json")
        return 0
    if args.output_dir is not None:
        root = _onboarding_target(args.output_dir)
        try:
            index = _write_onboarding(_config_requests(args), root)
        except (EOFError, KeyboardInterrupt):
            print("Setup cancelled.", file=sys.stderr)
            return 130
        print(
            "Requests and profiles saved. Runtime compatibility and FPM data are unchecked; "
            "setup has not launched GPU work."
        )
        print("Separate collection plan commands are ready; the same GPUs may be reused across separate runs.")
        _print({"onboarding_index": str(root / "onboarding.json"), **index})
        return 0
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
    print(
        f"Scope: FPM simulation for {request.identity.model} ({request.identity.model_kind}), "
        f"vLLM {request.identity.framework_version}, "
        f"{request.identity.gpu}; {request.identity.interconnect}; "
        f"{request.parallel_preset} {request.parallelism()}; runtime context {request.search.context_length} tokens."
    )
    print("Collection settings: " + json.dumps(request.collection_settings(), sort_keys=True))
    print(f"Collection GPUs required: {request.worker_gpus} for one selected worker; availability is unchecked.")
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
            "collection_gpus_required": request.worker_gpus,
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
    if args.support_action == "validate-fpm":
        from .validation import run_validation

        return run_validation(args)
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
