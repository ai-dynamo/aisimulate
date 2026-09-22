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
    profile = request.profile_deployment()
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
                command.extend(("--" + name.replace("_", "-"), value))
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


def run_fpm(
    request: SupportRequest,
    *,
    output_dir: str | Path,
    execute: bool = False,
    smoke: bool = False,
    limit: int | None = None,
    resume: bool = False,
    checkpoint_dir: str | Path | None = None,
    deployment: FPMDeployment | None = None,
) -> int:
    """Preview without side effects; execution requires the matching saved plan."""

    from .plan import check_plan, plan_lock

    root = Path(output_dir).expanduser().resolve()
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
    with plan_lock(root):
        check_plan(request, root)
        _check_campaign_outputs(root, smoke=smoke, resume=resume, checkpoint_dir=checkpoint_dir)
        from collector.fpm_forward.cli import main as fpm_main

        # The collector reports input/plan failures through argparse before
        # entering run_resolved; only execution failures escape this call.
        try:
            return fpm_main(command[3:])
        except Exception as exc:
            print(f"aisimulate onboard collect-fpm failed: {exc}", file=sys.stderr)
            return 1
