# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the immutable local plan for one self-service support request."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any

import yaml

from aisimulate.output import prepare_output_directory

from .errors import SupportWorkflowError
from .identity import support_cell_id, support_cell_payload
from .schema import ExistingSupport, SupportRequest
from .search import bounded_topologies, recommendation_config

_ROLES = ("baseline", "top1", "top2", "top3")
_E2E_METRICS = ("ttft_ms", "tpot_ms", "output_throughput_tok_s")


def existing_support(request: SupportRequest) -> ExistingSupport:
    """Read the packaged support matrix without treating a lookup failure as support."""

    try:
        from aiconfigurator.sdk.common import check_support

        support = check_support(
            request.identity.model,
            request.identity.gpu,
            request.identity.framework,
            request.identity.framework_version,
        )
    except Exception as exc:
        return ExistingSupport(status="unknown", detail=f"{type(exc).__name__}: {exc}")
    exact = support.exact_match
    aggregated = support.agg_supported if exact else False
    disaggregated = support.disagg_supported if exact else False
    requested = aggregated if request.identity.serving_mode == "aggregated" else disaggregated
    return ExistingSupport(
        status="supported" if requested else "unsupported",
        exact_match=exact,
        aggregated=aggregated,
        disaggregated=disaggregated,
        detail=(
            "a matching published support-matrix cell exists"
            if requested
            else (
                "architecture-level support is only advisory; no exact published model cell exists"
                if not exact
                else "the exact published model cell does not pass the requested mode; run this workflow"
            )
        ),
    )


def fpm_cli_args(
    request: SupportRequest,
    *,
    plan_only: bool,
    smoke: bool = False,
    limit: int | None = None,
    resume: bool = False,
    checkpoint_dir: str | None = None,
    artifact_root: str | None = None,
    database_root: str | None = None,
) -> list[str]:
    args = [
        "python3",
        "-m",
        "collector.fpm_forward",
        "--backend",
        request.fpm.backend,
        "--model-path",
        request.identity.model,
        "--gpu",
        request.identity.gpu,
        "--fpm-max-gpus",
        str(request.identity.gpu_count),
        "--fpm-gpu-counts",
        str(request.identity.gpu_count),
        "--fpm-parallel-presets",
        request.fpm.parallel_preset,
    ]
    if request.identity.sm is not None:
        args.extend(("--sm", str(request.identity.sm)))
    if plan_only:
        args.append("--plan-only")
    if smoke:
        args.append("--smoke")
    if limit is not None:
        args.extend(("--limit", str(limit)))
    if resume:
        args.append("--resume")
    if checkpoint_dir:
        args.extend(("--checkpoint-dir", checkpoint_dir))
    if artifact_root:
        args.extend(("--fpm-artifact-root", artifact_root))
    if database_root:
        args.extend(("--fpm-database-root", database_root))
    return args


def prepare_systems_overlay(request: SupportRequest, root: Path) -> Path:
    """Create the minimal systems tree consumed by the generated FPM configs."""

    source_root = Path(__file__).resolve().parents[2] / "aiconfigurator_core" / "systems"
    source_spec = source_root / f"{request.identity.gpu}.yaml"
    if not source_spec.is_file():
        raise ValueError(
            f"GPU {request.identity.gpu!r} has no packaged AIC system specification; "
            "the MVP can onboard new model/GPU cells but not new GPU architectures"
        )
    systems_root = root / "systems"
    systems_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_spec, systems_root / source_spec.name)
    (systems_root / "data").mkdir(exist_ok=True)
    return systems_root


def _commands(request: SupportRequest, root: Path) -> dict[str, Any]:
    fpm_paths = {
        "checkpoint_dir": str(root / "fpm-checkpoint"),
        "artifact_root": str(root / "fpm-artifacts"),
        "database_root": str(root / "systems" / "data"),
    }
    fpm_plan = fpm_cli_args(request, plan_only=True, **fpm_paths)
    fpm_run = fpm_cli_args(
        request,
        plan_only=False,
        **fpm_paths,
    )
    instance = request.execution.instance or "<BREV_INSTANCE>"
    return {
        "recommend": [
            [
                "aisimulate",
                "recommend",
                "--config",
                str(root / "recommend" / f"{workload.id}.yaml"),
                "--output-dir",
                str(root / "recommend-results" / workload.id),
                "--format",
                "json",
            ]
            for workload in request.workloads
        ],
        "fpm_plan_local": fpm_plan,
        "fpm_run_local": fpm_run,
        "fpm_plan_brev": ["brev", "exec", instance, shlex.join(fpm_plan)],
        "fpm_run_brev": ["brev", "exec", instance, shlex.join(fpm_run)],
        "validate": [
            "aisimulate",
            "support",
            "validate",
            "--config",
            str(root / "request.yaml"),
            "--evidence",
            str(root / "evidence.yaml"),
            "--output-dir",
            str(root / "validation"),
            "--systems-root",
            str(root / "systems"),
            "--format",
            "json",
        ],
    }


def _evidence_template(request: SupportRequest, cell_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for workload in request.workloads:
        for role in _ROLES:
            for metric in _E2E_METRICS:
                records.append(
                    {
                        "phase": "e2e",
                        "workload_id": workload.id,
                        "config_role": role,
                        "candidate_id": f"<REQUIRED:{workload.id}:{role}>",
                        "metric": metric,
                        "predicted": "<REQUIRED>",
                        "measured": "<REQUIRED>",
                        "gpu_count": request.identity.gpu_count,
                        "source_run_id": "<REQUIRED>",
                        "slo_compliant": "<REQUIRED:true|false>",
                        "held_out": False,
                    }
                )
    for workload in request.workloads:
        for role in _ROLES:
            for phase in ("fpm_prefill", "fpm_decode"):
                records.append(
                    {
                        "phase": phase,
                        "workload_id": None,
                        "config_role": role,
                        "candidate_id": f"<SAME-AS:{workload.id}:{role}>",
                        "metric": "forward_pass_ms",
                        "predicted": "<REQUIRED>",
                        "measured": "<REQUIRED>",
                        "gpu_count": request.identity.gpu_count,
                        "source_run_id": "<REQUIRED>",
                        "slo_compliant": None,
                        "held_out": True,
                    }
                )
    return {
        "schema_version": "aisimulate-support-evidence/v1",
        "support_cell_id": cell_id,
        "records": records,
    }


def create_plan(
    request: SupportRequest,
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if request.identity.framework != request.fpm.backend:
        raise SupportWorkflowError(
            f"the FPM MVP supports framework={request.fpm.backend!r}; "
            f"the requested cell uses {request.identity.framework!r}"
        )
    cell_id = support_cell_id(request)
    root = Path(output_dir)
    if overwrite and root.is_dir() and any(root.iterdir()):
        prior_plan = root / "support-plan.json"
        if not prior_plan.is_file():
            raise SupportWorkflowError(
                f"refusing to overwrite nonempty {root}: it is not a recognizable support-plan directory"
            )
        try:
            prior = json.loads(prior_plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SupportWorkflowError(f"could not verify the existing support plan in {root}: {exc}") from exc
        if not isinstance(prior, dict):
            raise SupportWorkflowError(f"existing support plan in {root} must contain one JSON object")
        prior_cell = prior.get("support_cell_id")
        if prior_cell != cell_id:
            raise SupportWorkflowError(
                f"refusing to mix support cells in {root}: existing={prior_cell!r}, requested={cell_id!r}; "
                "choose a new output directory"
            )
    root = prepare_output_directory(root, overwrite=overwrite)
    topologies = bounded_topologies(request)
    status = existing_support(request)
    systems_root = prepare_systems_overlay(request, root)

    (root / "recommend").mkdir(exist_ok=True)
    request_path = root / "request.yaml"
    request_path.write_text(
        yaml.safe_dump(request.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    for workload in request.workloads:
        path = root / "recommend" / f"{workload.id}.yaml"
        path.write_text(
            yaml.safe_dump(
                recommendation_config(
                    request,
                    workload,
                    topologies,
                    systems_path=str(systems_root),
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    commands = _commands(request, root)
    (root / "commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "evidence.yaml").write_text(
        yaml.safe_dump(_evidence_template(request, cell_id), sort_keys=False),
        encoding="utf-8",
    )

    plan = {
        "schema_version": "aisimulate-support-plan/v1",
        "support_cell_id": cell_id,
        "support_cell": support_cell_payload(request),
        "existing_support": status.model_dump(mode="json", exclude_none=True),
        "search": {
            "profile": request.search.version,
            "candidate_count": len(topologies),
            "max_candidates": request.search.max_candidates,
            "baseline_rule": "smallest legal candidate that completes without OOM",
            "candidates": [candidate.model_dump(mode="json") for candidate in topologies],
        },
        "fpm": {
            "status": "ready" if request.identity.framework == request.fpm.backend else "unsupported",
            "detail": (
                "whole-model FPM collection is available"
                if request.identity.framework == request.fpm.backend
                else f"FPM MVP supports {request.fpm.backend}; requested framework is {request.identity.framework}"
            ),
            "plan_command": commands["fpm_plan_local"],
        },
        "execution": request.execution.model_dump(mode="json", exclude_none=True),
        "outputs": {
            "request": str(request_path),
            "recommendation_configs": [
                str(root / "recommend" / f"{workload.id}.yaml") for workload in request.workloads
            ],
            "commands": str(root / "commands.json"),
            "evidence_template": str(root / "evidence.yaml"),
            "systems_root": str(systems_root),
        },
    }
    (root / "support-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan
