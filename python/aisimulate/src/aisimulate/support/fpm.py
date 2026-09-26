# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview or explicitly invoke the existing packaged FPM collector."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .schema import FPMDeployment, SupportRequest


def fpm_cli_args(
    request: SupportRequest,
    *,
    output_dir: str | Path,
    plan_only: bool,
    smoke: bool = False,
    limit: int | None = None,
    resume: bool = False,
    checkpoint_dir: str | Path | None = None,
    deployment: FPMDeployment | None = None,
) -> list[str]:
    if limit is not None and (not smoke or type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive cell count and requires smoke=True")
    root = Path(output_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else root / "fpm-checkpoint"
    if checkpoint != root / "fpm-checkpoint" and root / "fpm-checkpoint" not in checkpoint.parents:
        raise ValueError("checkpoint_dir must stay within the plan's fpm-checkpoint directory")
    if deployment is not None and deployment.executor == "slurm":
        from collector.fpm_forward.config import resolve_slurm_cpu_policy

        cpus, binding = resolve_slurm_cpu_policy(
            deployment.cpus_per_task,
            deployment.cpu_bind,
            resume=resume,
            checkpoint_dir=checkpoint,
            artifact_root=root / "fpm-artifacts",
            smoke=smoke,
        )
        deployment = deployment.model_copy(update={"cpus_per_task": cpus, "cpu_bind": binding})
    profile = request.profile_deployment()
    from .runtime import runtime_collection_inputs

    runtime_arguments, deployment = runtime_collection_inputs(request, deployment)
    scheduler = request.scheduler_limits()
    max_prefill_tokens = scheduler["max_batched_tokens"]
    max_prefill_batch = scheduler["max_sequences"]
    if max_prefill_tokens < 2:
        raise ValueError("FPM collection requires a rank-local token limit of at least 2 in the resource profile")
    command = [
        "python3",
        "-m",
        "collector.fpm_forward",
        "--backend",
        request.identity.framework,
        "--model-path",
        request.identity.model,
        "--gpu",
        request.identity.gpu,
        "--fpm-max-gpus",
        str(request.worker_gpus),
        "--fpm-gpu-counts",
        str(request.worker_gpus),
        "--fpm-parallel-presets",
        request.parallel_preset,
        "--fpm-max-model-len",
        str(request.search.context_length),
        "--fpm-max-num-batched-tokens",
        str(max_prefill_tokens),
        "--fpm-max-num-seqs",
        str(max_prefill_batch),
        "--fpm-max-prefill-isl",
        str(max_prefill_tokens),
        "--fpm-max-prefill-batch-size",
        str(max_prefill_batch),
        "--checkpoint-dir",
        str(checkpoint),
        "--fpm-artifact-root",
        str(root / "fpm-artifacts"),
        "--fpm-database-root",
        str(root / "systems/data"),
    ]
    command.extend(runtime_arguments)
    if "prefill_cudagraph_policy" in request.collection.model_fields_set:
        command.extend(("--fpm-prefill-cudagraph-policy", request.collection.prefill_cudagraph_policy))
    if request.collection.max_prefill_cudagraph_size is not None:
        command.extend(("--fpm-max-prefill-cudagraph-size", str(request.collection.max_prefill_cudagraph_size)))
    if request.collection.gpu_memory_utilization is not None:
        command.extend(("--fpm-gpu-memory-utilization", str(request.collection.gpu_memory_utilization)))
    if profile is not None:
        command.extend(
            (
                "--model-architecture",
                request.fpm_profile.architecture,
                "--fpm-weight-quantizations",
                profile.gemm_quant_mode,
                "--fpm-kv-cache-dtypes",
                profile.kv_cache_dtype,
                "--fpm-attention-backend",
                profile.attention_backend,
                "--fpm-moe-backend",
                profile.moe_backend,
                "--fpm-model-profile",
                str(root / "fpm-model-profile.json"),
            )
        )
    if request.identity.sm is not None:
        command.extend(("--sm", str(request.identity.sm)))
    if deployment is not None:
        if deployment.executor == "slurm":
            command.extend(("--fpm-executor", "slurm"))
        for name, value in deployment.model_dump(exclude_none=True, exclude={"executor", "container_mount"}).items():
            if name == "image":
                if deployment.executor == "slurm":
                    command.extend(("--fpm-slurm-container-image", value))
                else:
                    command.extend(("--generator-set", f"K8sConfig.k8s_image={json.dumps(value)}"))
            else:
                prefix = "--fpm-slurm-" if name in {"cpus_per_task", "cpu_bind"} else "--"
                command.extend((prefix + name.replace("_", "-"), str(value)))
        for mount in deployment.container_mount:
            command.extend(("--fpm-slurm-container-mount", mount))
    if plan_only:
        command.append("--plan-only")
    if smoke:
        command.append("--smoke")
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if resume:
        command.append("--resume")
    return command


def _check_campaign_outputs(root: Path, *, smoke: bool, resume: bool, checkpoint_dir: str | Path | None) -> None:
    checkpoint_name = "fpm_forward_smoke.json" if smoke else "fpm_forward.json"
    checkpoint_root = root / "fpm-checkpoint"
    artifact_paths = (root / "fpm-artifacts").glob("*/smoke/*" if smoke else "*/*")
    occupied = (
        any(smoke or path.name != "smoke" for path in artifact_paths)
        or any(checkpoint_root.rglob(checkpoint_name))
        or (not smoke and any((root / "systems/data").glob("*")))
    )
    if not occupied:
        return
    if not resume:
        raise ValueError(
            "existing campaign outputs require --resume with a matching checkpoint; "
            "choose a new output directory otherwise"
        )
    selected = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else checkpoint_root
    checkpoint = selected / checkpoint_name
    try:
        if checkpoint.is_symlink():
            raise ValueError("symlinked checkpoint")
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("schema"), str)
            or not payload["schema"]
            or not isinstance(payload.get("plan_sha256"), str)
            or len(payload["plan_sha256"]) != 64
            or not isinstance(payload.get("cells"), dict)
        ):
            raise ValueError("invalid checkpoint document")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"existing campaign outputs require a readable matching checkpoint at {checkpoint}; "
            "choose a new output directory otherwise"
        ) from exc
    # The collector remains responsible for the checkpoint schema and frozen-plan identity.


def _resolve_execution(command: list[str]):
    """Use the collector's public entry boundary, resolving the plan once."""
    from collector.fpm_forward.cli import _INPUT_ERRORS, _parser
    from collector.fpm_forward.entry import resolve_run_inputs
    from collector.model_cases import build_collection_case_plan

    parser = _parser()
    args = parser.parse_args(command[3:])
    try:
        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
        )
        resolved = resolve_run_inputs(args, case_plan)
    except _INPUT_ERRORS as error:
        parser.error(str(error))
    if args.smoke and args.limit is None:
        # Native smoke defaults to one cell. Onboarding must exercise decode
        # and every selected backend/precision cell, regardless of ordering.
        args.limit = len(resolved[0].cells)
    return args, resolved


def _checkpoint_root(root: Path, checkpoint_dir: str | Path | None) -> Path:
    selected = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else root / "fpm-checkpoint"
    if selected != root / "fpm-checkpoint" and root / "fpm-checkpoint" not in selected.parents:
        raise ValueError("checkpoint_dir must stay within the plan's fpm-checkpoint directory")
    return selected


def run_fpm(
    request: SupportRequest,
    *,
    output_dir: str | Path,
    execute: bool = False,
    check_readiness: bool = False,
    smoke: bool = False,
    limit: int | None = None,
    resume: bool = False,
    checkpoint_dir: str | Path | None = None,
    deployment: FPMDeployment | None = None,
) -> int:
    """Preview without side effects; execution requires the matching saved plan."""

    from .plan import check_plan, plan_lock

    root = Path(output_dir).expanduser().resolve()
    selected_checkpoint = _checkpoint_root(root, checkpoint_dir)
    if check_readiness:
        if execute or smoke or limit is not None or resume:
            raise ValueError("--check-readiness is read-only and cannot use --execute, --smoke, --limit or --resume")
        check_plan(request, root)
        from .collection_readiness import assess_readiness

        report = assess_readiness(request, root, selected_checkpoint)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready_for_full_collection"] else 1
    command = fpm_cli_args(
        request,
        output_dir=root,
        plan_only=not execute,
        smoke=smoke,
        limit=limit,
        resume=resume,
        checkpoint_dir=checkpoint_dir,
        deployment=deployment,
    )
    if not execute:
        if resume or (root.exists() and any(root.iterdir())):
            check_plan(request, root)
        print(shlex.join(command))
        return 0
    check_plan(request, root)
    from .runtime import runtime_probe_manifest, verify_collection_runtime, verify_runtime_acceptance

    verify_runtime_acceptance(request)
    with plan_lock(root):
        check_plan(request, root)
        _check_campaign_outputs(root, smoke=smoke, resume=resume, checkpoint_dir=checkpoint_dir)
        from collector.fpm_forward.entry import run_resolved
        from collector.fpm_forward.runner import _atomic_json

        from .collection_readiness import REPORT_FILENAME, assess_readiness, resume_without_workers

        frozen = None
        collector_status = None
        execution_error = None
        recovery_only = False
        try:
            args, resolved = _resolve_execution(command)
            frozen = resolved[0]
            if not smoke:
                before = assess_readiness(request, root, selected_checkpoint, expected_plan=frozen)
                recovery_only = resume and resume_without_workers(frozen, selected_checkpoint / "fpm_forward.json")
                if not before["ready_for_full_collection"] and not recovery_only:
                    raise ValueError(
                        "full collection requires usable prefill and decode readiness for this exact runtime/launch; "
                        "inspect --check-readiness and run --smoke --execute first"
                    )
            errors = run_resolved(args, resolved)
            status = collector_status = 1 if errors else 0
            if errors:
                print(json.dumps(errors, indent=2, sort_keys=True), file=sys.stderr)
            if status == 0 and not smoke and runtime_probe_manifest(request) is not None:
                payload = json.loads((selected_checkpoint / "fpm_forward.json").read_text(encoding="utf-8"))
                index = Path(payload["runtime_observations"])
                if not index.resolve().is_relative_to(root):
                    raise ValueError("formal runtime observation index must stay inside the collection directory")
                observed = verify_collection_runtime(request, index, collection_checkpoint=payload)
                (root / "runtime-compatibility.json").write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"aisimulate onboard collect-fpm failed: {exc}", file=sys.stderr)
            execution_error = str(exc)
            status = 1
        report = assess_readiness(request, root, selected_checkpoint, expected_plan=frozen)
        report.update(collector_exit_status=collector_status, recovery_only=recovery_only)
        if execution_error is not None:
            report["execution_error"] = execution_error
        _atomic_json(root / REPORT_FILENAME, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return status or (0 if report["ready_for_full_collection"] else 1)
