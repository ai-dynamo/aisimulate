# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind data labels to separately observed collection-runtime versions."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from .schema import FPMDeployment, SupportRequest

_BINDING = "runtime-version.json"
_MARKER = "AISIMULATE_RUNTIME_VERSION="
_PROBE = """import importlib.metadata as m, json
result = {"backend_version": m.version("vllm")}
try:
    result["dynamo_version"] = m.version("ai-dynamo")
except m.PackageNotFoundError:
    pass
print("AISIMULATE_RUNTIME_VERSION=" + json.dumps(result, sort_keys=True))
"""


def _run(command: list[str], *, payload: str | None = None) -> str:
    try:
        result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=300, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        diagnostic = getattr(exc, "stderr", None) or str(exc)
        raise ValueError(f"collection image runtime version detection failed: {diagnostic}") from exc
    return result.stdout


def detect_runtime_versions(deployment: FPMDeployment) -> dict[str, str]:
    """Read package metadata in the target image without loading a model/GPU."""
    if deployment.image is None:
        raise ValueError("runtime version detection requires --image for the actual collection container")
    if deployment.executor == "slurm":
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("runtime version detection requires a caller-owned Slurm allocation")
        command = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--gpus=0",
            "--overlap",
            f"--container-image={deployment.image}",
        ]
        if deployment.container_mount:
            command.append("--container-mounts=" + ",".join(deployment.container_mount))
        output = _run([*command, "python3", "-c", _PROBE])
    else:
        name = "ais-fpm-version-" + uuid.uuid4().hex[:12]
        namespace = deployment.namespace or "default"
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "restartPolicy": "Never",
                "activeDeadlineSeconds": 240,
                "containers": [
                    {
                        "name": "probe",
                        "image": deployment.image,
                        "command": ["python3", "-c", _PROBE],
                        "resources": {"requests": {"cpu": "1", "memory": "128Mi"}},
                    }
                ],
            },
        }
        if deployment.image_pull_secret:
            manifest["spec"]["imagePullSecrets"] = [{"name": deployment.image_pull_secret}]
        prefix = ["kubectl", "--namespace", namespace]
        created = False
        try:
            _run([*prefix, "create", "-f", "-"], payload=json.dumps(manifest))
            created = True
            # Waiting for termination covers both success and a missing package.
            _run(
                [
                    *prefix,
                    "wait",
                    f"pod/{name}",
                    "--for=jsonpath={.status.containerStatuses[0].state.terminated.exitCode}",
                    "--timeout=240s",
                ]
            )
            output = _run([*prefix, "logs", name])
            status = json.loads(_run([*prefix, "get", "pod", name, "-o", "json"]))
            terminated = status["status"]["containerStatuses"][0]["state"]["terminated"]
            if terminated["exitCode"] != 0:
                raise ValueError(f"collection image runtime version detection failed: {output.strip()}")
        finally:
            if created:
                _run([*prefix, "delete", "pod", name, "--ignore-not-found", "--wait=false"])
    lines = [line[len(_MARKER) :] for line in output.splitlines() if line.startswith(_MARKER)]
    if len(lines) != 1:
        raise ValueError("collection image did not report one runtime version; no local/latest fallback is allowed")
    result = json.loads(lines[0])
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("backend_version"), str)
        or not result["backend_version"].strip()
    ):
        raise ValueError("collection image reported an empty or invalid backend version")
    return result


def _deployment_identity(deployment: FPMDeployment | None) -> dict:
    return (deployment or FPMDeployment()).model_dump(mode="json", exclude_none=True, exclude={"dynamo_version"})


def runtime_deployment(deployment: FPMDeployment | None, output_dir: str | Path) -> FPMDeployment | None:
    """Reuse frozen launch inputs, including when addressed by the child path."""
    from .plan import request_id

    root = Path(output_dir).expanduser().resolve()
    binding = root / _BINDING
    child_path = False
    if not binding.exists() and root.name == "runtime-resolved":
        binding = root.parent / _BINDING
        child_path = True
    if not binding.exists():
        return deployment
    if binding.is_symlink():
        raise ValueError("refusing a symlinked runtime version binding")
    value = json.loads(binding.read_text())
    if child_path:
        child = SupportRequest.from_yaml(root / "request.yaml")
        if request_id(child) != value.get("resolved_request_id"):
            raise ValueError("resolved runtime request differs from its deployment binding")
    frozen = FPMDeployment.model_validate(value["deployment"])
    if deployment is not None:
        selected = deployment.model_dump(exclude_none=True, exclude_defaults=True, exclude={"dynamo_version"})
        for key, item in selected.items():
            if frozen.model_dump()[key] != item:
                raise ValueError("collection deployment differs from the runtime version probe; use a fresh plan")
    if deployment is not None and deployment.dynamo_version is not None:
        frozen = frozen.model_copy(update={"dynamo_version": deployment.dynamo_version})
    return frozen


def bound_runtime_request(
    request: SupportRequest,
    deployment: FPMDeployment | None,
    output_dir: str | Path,
) -> tuple[SupportRequest, Path] | None:
    """Reuse only an intact child bound to this registration and deployment."""
    from .plan import check_plan, request_id

    root = Path(output_dir).expanduser().resolve()
    binding = root / _BINDING
    if not binding.exists():
        return None
    if binding.is_symlink():
        raise ValueError("refusing a symlinked runtime version binding")
    check_plan(request, root)
    value = json.loads(binding.read_text())
    if value.get("source_request_id") != request_id(request):
        raise ValueError("runtime version binding belongs to a different registration")
    if deployment is not None and value.get("deployment") != _deployment_identity(deployment):
        raise ValueError("collection deployment differs from the runtime version probe; use a fresh plan")
    resolved_root = root / "runtime-resolved"
    if resolved_root.is_symlink():
        raise ValueError("refusing a symlinked resolved runtime plan")
    resolved = SupportRequest.from_yaml(resolved_root / "request.yaml")
    if request_id(resolved) != value.get("resolved_request_id"):
        raise ValueError("resolved runtime request differs from its version binding")
    check_plan(resolved, resolved_root)
    return resolved, resolved_root


def resolve_runtime_request(
    request: SupportRequest,
    deployment: FPMDeployment | None,
    output_dir: str | Path,
) -> tuple[SupportRequest, Path]:
    """Observe runtime, then freeze a separate plan; never rewrite registration."""
    from .plan import check_plan, create_plan, plan_lock, request_id

    root = Path(output_dir).expanduser().resolve()
    existing = bound_runtime_request(request, deployment, root)
    if existing is not None:
        return existing
    deployment = deployment or FPMDeployment()
    with plan_lock(root):
        existing = bound_runtime_request(request, deployment, root)
        if existing is not None:
            return existing
        check_plan(request, root)
        # A legacy in-progress campaign cannot acquire new identity silently.
        if any((root / name).exists() and any((root / name).iterdir()) for name in ("fpm-checkpoint", "fpm-artifacts")):
            raise ValueError("existing collection artifacts require their original runtime-bound plan")
        observed = detect_runtime_versions(deployment)
        actual = observed.get("backend_version")
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError("target runtime backend version detection returned no version")
        payload = request.model_dump(mode="json")
        label = request.identity.framework_version or actual
        payload["identity"].update(framework_version=label, runtime_framework_version=actual)
        if payload.get("fpm_profile"):
            profile = payload["fpm_profile"]
            for item in profile["deployments"]:
                if item["backend_version"] is None:
                    item["backend_version"] = label
            # Only update our structured derivation provenance; opaque user
            # provenance remains authoritative and is never discarded.
            for holder in [profile, *(item["resources"] for item in profile["deployments"])]:
                try:
                    provenance = json.loads(holder["provenance"])
                except (TypeError, ValueError):
                    continue
                if isinstance(provenance, dict) and "deployment_identity" in provenance:
                    provenance["deployment_identity"] = payload["identity"]
                    holder["provenance"] = json.dumps(provenance, sort_keys=True)
        resolved = SupportRequest.model_validate(payload)
        resolved_root = root / "runtime-resolved"
        # Interrupted creation is recoverable only if all existing child bytes
        # match this same fully resolved request (create_plan verifies them).
        create_plan(resolved, resolved_root, overwrite=resolved_root.exists())
        record = {
            "schema_version": 1,
            "source_request_id": request_id(request),
            "resolved_request_id": request_id(resolved),
            "deployment": _deployment_identity(deployment),
            "runtime": observed,
            "dynamo_version": deployment.dynamo_version or observed.get("dynamo_version"),
        }
        binding = root / _BINDING
        if binding.is_symlink():
            raise ValueError("refusing a symlinked runtime version binding")
        temporary = root / (".runtime-version-" + uuid.uuid4().hex + ".json")
        try:
            temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            temporary.replace(binding)
        finally:
            temporary.unlink(missing_ok=True)
    return resolved, resolved_root
