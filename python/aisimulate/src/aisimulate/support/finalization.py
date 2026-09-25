# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finalize memory from an immutable collection into a fresh simulation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from aisimulate.fpm_profile import FpmResourceProfile

from .plan import _plan_documents, check_plan, plan_lock, request_id
from .schema import FPMDeployment, SupportRequest

_SCHEMA = "aisimulate-onboarding-finalization/v1"


def add_finalization_parser(actions: Any) -> None:
    parser = actions.add_parser(
        "finalize",
        help="Resolve memory from collected runtime evidence and create fresh simulation inputs.",
        description=(
            "Verify a completed formal FPM collection, resolve its runtime cache allocation, and write a fresh "
            "request/profile, predict/recommend configs and a verified copy of the timing data. "
            "The original collection is unchanged. Review the resolved profile before accepting it in your "
            "onboarding checkpoint. No GPU work is launched."
        ),
    )
    parser.add_argument("-c", "--config", required=True, help="Original reviewed onboarding request.")
    parser.add_argument("--output-dir", required=True, help="Original completed collection plan directory.")
    parser.add_argument("--resolved-output-dir", required=True, help="Fresh directory for resolved simulation inputs.")
    parser.add_argument(
        "--memory-config",
        help="Accepted capacity-only request revision imported from this collection's formal observations.",
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _inside(path: Path, parent: Path) -> Path:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent):
        raise ValueError(f"expected a regular collection artifact inside {parent}: {path}")
    for directory in path.parents:
        if directory == parent:
            break
        if directory.is_symlink():
            raise ValueError(f"refusing symlinked collection artifact: {path}")
    return path


def finalization_manifest(request: SupportRequest) -> dict[str, Any] | None:
    """Return an onboarding-owned manifest, without interpreting other provenance."""
    deployment = request.profile_deployment()
    runtime = deployment.resources.runtime_memory if deployment is not None else None
    if runtime is None:
        return None
    try:
        value = json.loads(runtime.provenance)
    except ValueError:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != _SCHEMA:
        return None
    if "observation_provenance" in value and value["observation_provenance"] != "source_references":
        raise ValueError("unsupported finalized observation provenance format")
    return value


def verify_finalized_data(request: SupportRequest, root: Path) -> None:
    manifest = finalization_manifest(request)
    if manifest is None:
        return
    artifacts = manifest.get("formal_data")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("finalized data manifest requires the verified formal data pair")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("relative_path"), str):
            raise ValueError("finalized data manifest contains an invalid artifact")
        relative = Path(artifact["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("systems", "data"):
            raise ValueError("finalized data manifest contains an invalid relative path")
        path = _inside(root / relative, root / "systems/data")
        if _digest(path.read_bytes()) != artifact.get("sha256"):
            raise ValueError(f"finalized FPM data changed: {path}")


def _merge_resources(
    observations: list[dict[str, Any]], manifest: dict[str, Any], *, source_references: bool = False
) -> dict[str, Any]:
    if not observations:
        raise ValueError("collection has no runtime memory observations")
    canonical = None
    runtime_settings = None
    runtime_geometry = None
    minimum = None
    summaries = []
    for observation in observations:
        resources = FpmResourceProfile.model_validate(observation)
        if resources.memory_source != "runtime" or resources.cache_layout != "grouped":
            raise ValueError("finalization requires observed grouped runtime cache resources")
        evidence = json.loads(resources.runtime_memory.provenance)
        settings = evidence.get("runtime_settings") if isinstance(evidence, dict) else None
        if not isinstance(settings, dict) or not settings:
            raise ValueError("runtime memory evidence lacks resolved launch settings")
        if runtime_settings is None:
            runtime_settings = settings
        elif settings != runtime_settings:
            raise ValueError(
                "collection cells have incompatible resolved launch settings, including graph configuration; "
                "collect memory with the intended serving configuration"
            )
        geometry = next(
            (
                item["evidence"]["cache"]["groups"]
                for item in evidence["artifacts"]
                if item["evidence"]["kind"] == "worker"
            ),
            None,
        )
        if not geometry:
            raise ValueError("runtime memory evidence lacks physical cache group geometry")
        if runtime_geometry is None:
            runtime_geometry = geometry
        elif geometry != runtime_geometry:
            raise ValueError("collection cells have incompatible runtime cache layouts or layer groups")
        comparable = resources.model_dump(mode="json")
        comparable.pop("provenance")
        runtime = comparable["runtime_memory"]
        runtime.pop("provenance")
        capacity = runtime.pop("kv_cache_bytes")
        if canonical is None:
            canonical = comparable
        elif comparable != canonical:
            raise ValueError("collection cells have incompatible runtime cache layouts or settings")
        minimum = capacity if minimum is None else min(minimum, capacity)
        if source_references:
            # Raw records remain immutable collection artifacts. Reference their
            # exact bytes instead of copying every rank/layer into each candidate.
            evidence["artifacts"] = [
                {key: value for key, value in artifact.items() if key != "evidence"}
                for artifact in evidence["artifacts"]
            ]
            summaries.append(
                {
                    **observation,
                    "runtime_memory": {**observation["runtime_memory"], "provenance": _canonical(evidence)},
                }
            )
    assert canonical is not None and minimum is not None
    if source_references:
        manifest["observation_provenance"] = "source_references"
    manifest["cell_observations"] = summaries if source_references else observations
    manifest["capacity_policy"] = "minimum observed request-usable rank-local capacity across compatible cells"
    provenance = _canonical(manifest)
    return {
        **canonical,
        "runtime_memory": {**canonical["runtime_memory"], "kv_cache_bytes": minimum, "provenance": provenance},
        "provenance": (
            "Runtime cache resources resolved from verified collection evidence. Raw records are retained at "
            "the SHA-256-bound sources in runtime_memory.provenance."
            if source_references
            else provenance
        ),
    }


def _verify_collection(
    request: SupportRequest, root: Path, *, memory_revision: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[Path, bytes], dict[Path, str]]:
    from collector.fpm_forward.cli import _parser
    from collector.fpm_forward.config import FPMCollectionOptions
    from collector.fpm_forward.database import aggregate_cell, validate_formal_database_commit
    from collector.fpm_forward.planner import backend_identity_columns
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA
    from collector.fpm_forward.runtime_memory import (
        cell_from_dict,
        resolve_runtime_resources,
        saved_plan_identity,
    )

    deployment = request.profile_deployment()
    if deployment is None:
        raise ValueError("finalization requires the original FPM profile used for collection")
    paths = list((root / "fpm-checkpoint").rglob("fpm_forward.json"))
    if len(paths) != 1:
        raise ValueError("finalization requires exactly one formal collector checkpoint under fpm-checkpoint")
    checkpoint_path = _inside(paths[0], root)
    checkpoint = _json(checkpoint_path)
    sha = checkpoint.get("plan_sha256")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("invalid formal collector checkpoint identity")
    collection_root = root / "fpm-artifacts" / sha[:16]
    collection_path = _inside(collection_root / "collection-plan.json", root)
    payload = _json(collection_path)
    frozen = saved_plan_identity(payload)
    if frozen.sha256 != sha:
        raise ValueError("formal collector checkpoint differs from the saved collection plan")
    expected = {
        "model_path": request.identity.model,
        "system": request.identity.gpu,
        "backend": request.identity.framework,
        "fpm_profile": request.fpm_profile.model_dump(mode="json"),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("saved collection plan differs from the original onboarding profile or identity")
    if frozen.capability.aic_database_version != request.identity.framework_version:
        raise ValueError("saved collection runtime version differs from onboarding")
    scheduler = request.scheduler_limits()
    from .fpm import fpm_cli_args
    from .runtime import runtime_probe_manifest, verify_collection_runtime

    probe_manifest = runtime_probe_manifest(request)
    memory_request = _memory_revision_request(memory_revision) if memory_revision is not None else None
    if memory_request is not None and probe_manifest is None:
        raise ValueError("memory revision requires the original imported runtime profile")

    # Normalize the reviewed inputs through the same local collector options
    # parser. This does not resolve a model or regenerate Dynamo's runtime grid.
    saved_options = dict(payload["options"])
    # Deployment is selected at collection, after the request is saved. Retain
    # its verified frozen values while deriving runtime/grid limits from the
    # reviewed request. Parsing these values never invokes either executor.
    collection_deployment = FPMDeployment(
        executor=saved_options.get("executor", "kubernetes"),
        image=saved_options.get("slurm_container_image") or None,
        container_mount=saved_options.get("slurm_container_mounts", []),
    )
    if probe_manifest is not None:
        source_index = _json(Path(probe_manifest["observations_index"]))
        collection_deployment = FPMDeployment.model_validate(
            source_index["configurations"][probe_manifest["configuration"]]["launch"]["deployment"]
        )
    reviewed_options = FPMCollectionOptions.from_args(
        _parser().parse_args(
            fpm_cli_args(request, output_dir=root, plan_only=True, deployment=collection_deployment)[3:]
        )
    ).to_dict()
    for options in (reviewed_options, saved_options):
        options.setdefault("gpu_memory_utilization", request.collection.memory_fraction)
    if saved_options != reviewed_options:
        raise ValueError("saved collection options differ from the reviewed runtime and collection settings")
    cells = [cell_from_dict(value) for value in payload["cells"]]
    if not cells or {cell.workload_kind for cell in cells} != {"prefill", "decode"}:
        raise ValueError("finalization requires complete prefill and decode collection")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("saved collection has duplicate cell identities")
    database = checkpoint.get("database", {})
    if (
        not isinstance(database, dict)
        or database.get("status") != "passed"
        or database.get("missing_cells") != []
        or database.get("skipped_first_publisher_wins") != []
        or database.get("plan_cells") != len(cells)
        or database.get("published_cells") != len(cells)
    ):
        raise ValueError("finalization requires complete formal publication without missing or reused cells")
    if any(not isinstance(database.get(name), str) for name in ("parquet", "metadata")):
        raise ValueError("formal collector checkpoint is missing its published data paths")
    parquet = _inside(Path(database["parquet"]), root / "systems/data")
    metadata = _inside(Path(database["metadata"]), root / "systems/data")
    if parquet.name != "fpm_forward_perf.parquet" or metadata != parquet.with_name("fpm_forward_perf.metadata.json"):
        raise ValueError("collector checkpoint does not reference a canonical formal FPM pair")
    validate_formal_database_commit(parquet, metadata, frozen)
    snapshots = {path: _digest(path.read_bytes()) for path in (checkpoint_path, collection_path, parquet, metadata)}
    rows, observations = [], []
    expected_topology = {
        "tp": deployment.tp,
        "pp": deployment.pp,
        "dp": deployment.dp,
        "moe_tp": deployment.moe_tp,
        "moe_ep": deployment.moe_ep,
        "cp": deployment.cp,
    }
    if not isinstance(checkpoint.get("cells"), dict):
        raise ValueError("formal collector checkpoint is missing successful cell attempts")
    for cell in cells:
        if any(getattr(cell.topology, key) != value for key, value in expected_topology.items()):
            raise ValueError("saved collection topology differs from the reviewed deployment")
        for name in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype"):
            if getattr(cell, name) != getattr(deployment, name):
                raise ValueError(f"saved collection {name} differs from the reviewed deployment")
        for name, value in backend_identity_columns(cell.backend_policy).items():
            if value != getattr(deployment, name):
                raise ValueError(f"saved collection {name} differs from the reviewed deployment")
        entry = checkpoint.get("cells", {}).get(cell.cell_id, {})
        attempt = entry.get("attempt_id") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or entry.get("status") != "passed"
            or not isinstance(attempt, str)
            or not attempt
        ):
            raise ValueError(f"collection cell {cell.cell_id} has no successful attempt")
        cell_dir = collection_root / "cells" / cell.cell_id
        if _json(_inside(cell_dir / "cell.json", root)) != cell.to_dict():
            raise ValueError(f"saved cell differs from frozen plan: {cell.cell_id}")
        for path in cell_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"refusing symlinked collection artifact: {path}")
            if path.is_file():
                snapshots[path] = _digest(path.read_bytes())
        rows.extend(aggregate_cell(frozen, cell, cell_dir, expected_attempt_id=attempt))
        if probe_manifest is None:
            observations.append(
                resolve_runtime_resources(
                    cell,
                    cell_dir / "raw",
                    expected_plan_sha256=sha,
                    expected_attempt_id=attempt,
                    expected_backend_version=request.identity.framework_version,
                    expected_context_length=request.search.context_length,
                    expected_max_num_tokens=scheduler["max_batched_tokens"],
                    expected_max_batch_size=scheduler["max_sequences"],
                    expected_gpu_memory_utilization=request.collection.memory_fraction,
                    expected_model_revision=request.identity.model_revision,
                )
            )
    runtime_compatibility = None
    if probe_manifest is not None:
        if not isinstance(checkpoint.get("runtime_observations"), str):
            raise ValueError("formal collection is missing its runtime observation index")
        index = _inside(Path(checkpoint["runtime_observations"]), root)
        observed = verify_collection_runtime(
            request, index, collection_checkpoint=checkpoint, memory_request=memory_request
        )
        observations.append(observed["resources"])
        runtime_compatibility = observed["compatibility"]
        snapshots[index] = _digest(index.read_bytes())
        evidence = observed["provenance"]
        for reference in [*evidence["launch_artifacts"], *evidence["artifacts"], *evidence["runtime_artifacts"]]:
            path = _inside(index.parent / reference["path"], root)
            snapshots[path] = _digest(path.read_bytes())
        from collector.fpm_forward.runtime_instrumentation import load_instrumentation

        bundle_path = _inside(index.parent / evidence["instrumentation"]["manifest"], root)
        bundle = load_instrumentation(bundle_path)
        for path in [bundle_path, *(bundle.root / relative for relative in bundle.files)]:
            snapshots[_inside(path, root)] = _digest(path.read_bytes())
    import pyarrow.parquet as pq

    actual = pq.read_table(parquet).to_pylist()
    if sorted(map(_canonical, actual)) != sorted(map(_canonical, rows)):
        raise ValueError("formal FPM rows differ from the verified native collection or include unrelated cells")
    files = {path.relative_to(root): path.read_bytes() for path in (parquet, metadata)}
    manifest = {
        "schema_version": _SCHEMA,
        "source_request_id": request_id(request),
        "source_collection_plan_sha256": sha,
        "source_directory": str(root),
        "review_status": "requires_review",
        "modeling_scope": (
            "Text-decoder timing; observed cache capacity accounts for every component loaded by the benchmark "
            "worker. No guessed multimodal encoder allocation is subtracted."
        ),
        "source_artifacts": [{"path": str(path), "sha256": digest} for path, digest in sorted(snapshots.items())],
        "formal_data": [
            {"relative_path": str(path), "sha256": _digest(content)} for path, content in sorted(files.items())
        ],
    }
    if runtime_compatibility is not None:
        manifest["runtime_probe"] = probe_manifest
        manifest["runtime_compatibility"] = runtime_compatibility
    if memory_revision is not None:
        manifest["memory_revision"] = memory_revision
    return observations, manifest, files, snapshots


def _memory_revision_request(reference: dict[str, Any]) -> SupportRequest:
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("invalid memory revision reference")
    path = Path(reference["path"])
    if path.is_symlink() or not path.is_file() or _digest(path.read_bytes()) != reference.get("sha256"):
        raise ValueError(f"memory revision request changed: {path}")
    request = SupportRequest.from_yaml(path)
    if request_id(request) != reference.get("request_id"):
        raise ValueError("memory revision request identity changed")
    return request


def finalize(
    request: SupportRequest,
    output_dir: str | Path,
    resolved_output_dir: str | Path,
    *,
    memory_config: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    raw_target = Path(resolved_output_dir).expanduser()
    if raw_target.is_symlink():
        raise ValueError("resolved output must not be a symlink")
    target = raw_target.resolve()
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("resolved output must be separate from the original collection directory")
    if target.exists():
        raise ValueError("resolved output directory must be new; existing artifacts cannot be replaced")
    with plan_lock(root):
        check_plan(request, root)
        memory_revision = None
        if memory_config is not None:
            from .runtime import verify_runtime_acceptance

            path = Path(memory_config).expanduser().absolute()
            revised = SupportRequest.from_yaml(path)
            memory_revision = {
                "path": str(path),
                "sha256": _digest(path.read_bytes()),
                "request_id": request_id(revised),
            }
            _memory_revision_request(memory_revision)
            verify_runtime_acceptance(revised)
        observations, manifest, formal_files, snapshots = _verify_collection(
            request, root, memory_revision=memory_revision
        )
        resources = _merge_resources(observations, manifest, source_references=True)
        payload = request.model_dump(mode="json")
        payload["fpm_profile"]["provenance"] = _canonical(
            {
                "memory_source": "runtime",
                "method": "runtime memory resolved from verified collector artifacts",
                "source_collection_plan_sha256": manifest["source_collection_plan_sha256"],
                "source_profile_provenance": request.fpm_profile.provenance,
            }
        )
        selected = request.profile_deployment()
        for index, deployment in enumerate(request.fpm_profile.deployments):
            if deployment == selected:
                payload["fpm_profile"]["deployments"][index]["resources"] = resources
        resolved = SupportRequest.model_validate(payload)
        plan, documents = _plan_documents(resolved, target)
        documents.update(formal_files)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            for relative, content in documents.items():
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            check_plan(request, root)
            for path, digest in snapshots.items():
                if _digest(_inside(path, root).read_bytes()) != digest:
                    raise ValueError(f"collection artifact changed during finalization: {path}")
            if memory_revision is not None:
                _memory_revision_request(memory_revision)
            if target.exists():
                raise ValueError("resolved output appeared during finalization; refusing to replace it")
            os.rename(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {
        "resolved_directory": str(target),
        "request": plan["outputs"]["request"],
        "review_status": "requires_review",
    }


def run_finalization(args: argparse.Namespace) -> int:
    result = finalize(
        SupportRequest.from_yaml(args.config),
        args.output_dir,
        args.resolved_output_dir,
        memory_config=args.memory_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Runtime memory resolved. Review the new profile before accepting it in the onboarding checkpoint.")
    return 0
