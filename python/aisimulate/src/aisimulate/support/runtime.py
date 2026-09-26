# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed runtime probing and CPU-only observation import."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from aisimulate.fpm_profile import FpmModelProfile

from .config_profile import load_model_config, profile_from_observations, validate_overrides
from .schema import (
    AGENTX_REFERENCE_CONTEXT,
    CollectionSpec,
    FPMDeployment,
    SearchProfile,
    SupportIdentity,
    SupportRequest,
)

PROVENANCE_SCHEMA = "aisimulate-onboarding-runtime-probe/v1"
_PRECISION = (
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "kv_cache_dtype",
    "comm_quant_mode",
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
)
_TOPOLOGY = ("tensor_parallel", "attention_data_parallel", "moe_tensor_parallel", "moe_expert_parallel")


def add_runtime_parsers(actions: Any) -> None:
    from .cli import add_deployment_arguments

    probe = actions.add_parser(
        "probe-runtime", help="Preview or execute cache observations before profile geometry is known."
    )
    imported = actions.add_parser(
        "import-observations", help="Validate saved runtime evidence into fresh unaccepted profile drafts."
    )
    for parser in (probe, imported):
        parser.add_argument(
            "--checkpoint", required=True, help="Existing single onboarding checkpoint, outside output directories."
        )
        parser.add_argument(
            "--configuration", action="append", help="Existing checkpoint key; repeat or omit for every configuration."
        )
        parser.add_argument("--output-dir", required=True)
    probe.add_argument(
        "--instrumentation", help="Campaign-local YAML/JSON manifest; otherwise use a compatible bundled observer."
    )
    probe.add_argument("--execute", action="store_true", help="Execute both phases for every selected configuration.")
    probe.add_argument("--resume", action="store_true")
    add_deployment_arguments(probe, default_executor=None)
    imported.add_argument(
        "--observations", required=True, help="Runner-owned observations.json index; observer code is never executed."
    )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _inputs(state: Any, config: Any) -> dict[str, Any]:
    from .checkpoint import _merge

    return _merge(state.inputs, config.inputs)


def checkpoint_launch(
    state: Any,
    name: str,
    checkpoint: Path,
    deployment_overrides: dict[str, Any] | None = None,
    *,
    resolve_cpu_defaults: bool = False,
    resume_output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve stage-2 facts; configuration names never determine topology."""
    from collector.fpm_forward.runtime_probe import normalize_probe_launch

    config = state.configurations[name]
    inputs = _inputs(state, config)
    payload = deepcopy(config.draft_request)
    for section, model in (("identity", SupportIdentity), ("search", SearchProfile), ("collection", CollectionSpec)):
        values = dict(_object(payload.get(section, {}), f"draft_request.{section}"))
        for field in model.model_fields:
            if field in inputs:
                if field in values and values[field] != inputs[field]:
                    raise ValueError(f"inputs.{field} conflicts with draft_request.{section}.{field}")
                values[field] = inputs[field]
        payload[section] = values
    if not config.draft_request:
        payload["collection"].setdefault("prefill_cudagraph_policy", "runtime")
    if not payload.get("fpm_profile"):
        missing = [key for key in (*_TOPOLOGY, "context_length") if payload["search"].get(key) is None]
        if missing:
            raise ValueError("explicit search launch facts required: " + ", ".join(missing))
    # A probe validates launch selections, independently of a simulation
    # workload or the previous profile's observed resource envelope.
    identity = SupportIdentity.model_validate(payload["identity"])
    profile = FpmModelProfile.model_validate(payload["fpm_profile"]) if payload.get("fpm_profile") else None
    search_values = dict(payload["search"])
    if profile is not None:
        search_values.setdefault("context_length", min(profile.context_length, AGENTX_REFERENCE_CONTEXT))
    search = SearchProfile.model_validate(search_values)
    collection = CollectionSpec.model_validate(payload["collection"])
    topology = {
        "tp": search.tensor_parallel,
        "pp": 1,
        "dp": search.attention_data_parallel or 1,
        "moe_tp": search.moe_tensor_parallel or (search.tensor_parallel if identity.model_kind == "moe" else 1),
        "moe_ep": search.moe_expert_parallel or 1,
        "cp": 1,
    }
    precision = {}
    selected = (
        profile.select(
            model=identity.model,
            system=identity.gpu,
            backend=identity.framework,
            backend_version=identity.framework_version,
            tp_size=topology["tp"],
            attention_dp_size=topology["dp"],
            moe_tp_size=topology["moe_tp"],
            moe_ep_size=topology["moe_ep"],
        )
        if profile is not None
        else None
    )
    if selected is not None:
        precision.update({name: getattr(selected, name) for name in _PRECISION})
    resource_overrides = validate_overrides(inputs.get("resource_overrides"))
    provided = {key: value for key, value in resource_overrides.items() if key in _PRECISION}
    explicit = _object(inputs.get("precision", {}), "inputs.precision")
    if set(explicit) - set(_PRECISION):
        raise ValueError("unknown inputs.precision fields: " + ", ".join(sorted(set(explicit) - set(_PRECISION))))
    for key, value in explicit.items():
        if key in provided and provided[key] != value:
            raise ValueError(f"inputs.precision.{key} conflicts with resource_overrides")
        provided[key] = value
    for key, value in provided.items():
        if key in precision and precision[key] != value:
            raise ValueError(f"inputs.precision.{key} conflicts with draft profile; edit the draft before probing")
        precision[key] = value
    precision.setdefault("comm_quant_mode", "half")
    if "kv_cache_dtype" in precision:
        precision["kvcache_quant_mode"] = precision.pop("kv_cache_dtype")
    model = inputs.get("model_config")
    if isinstance(model, str):
        model = {"path": model}
    model = _object(model, "inputs.model_config (path string or path/sha256 object)")
    path = Path(model.get("path", "")).expanduser()
    # HF cache snapshots use symlinks into a blob directory. Keep the logical
    # parent so adjacent configuration sources are read from that snapshot.
    path = (path if path.is_absolute() else checkpoint.parent / path).absolute()
    model_config = load_model_config(path)
    if model.get("sha256") is not None and model["sha256"] != model_config.sha256:
        raise ValueError("model config content differs from inputs.model_config.sha256")
    # The collector's explicit local config loader reads this adjacent ModelOpt
    # sidecar. Preserve its bytes separately from the unmodified config.json.
    sidecar = path.parent / "hf_quant_config.json"
    source_files = {sidecar.name: _hash(sidecar)} if sidecar.is_file() else {}
    if model.get("source_files", source_files) != source_files:
        raise ValueError("model config sidecar files differ from inputs.model_config.source_files")
    deployment = {
        **_object(inputs.get("collection_deployment", {}), "inputs.collection_deployment"),
        **(deployment_overrides or {}),
    }
    deployment = FPMDeployment.model_validate(deployment).model_dump(mode="json", exclude_none=True)
    if resume_output_dir is not None:
        from collector.fpm_forward.runtime_probe import resolve_probe_cpu_policy

        deployment = resolve_probe_cpu_policy(name, deployment, resume_output_dir, resume=True)
    elif resolve_cpu_defaults and deployment["executor"] == "slurm":
        from collector.fpm_forward.config import resolve_slurm_cpu_policy

        cpus, binding = resolve_slurm_cpu_policy(deployment.get("cpus_per_task"), deployment.get("cpu_bind"))
        deployment.update(cpus_per_task=cpus, cpu_bind=binding)
    max_tokens = collection.max_num_tokens or (selected.resources.max_num_tokens if selected else 8192)
    max_sequences = collection.max_batch_size or (selected.resources.max_batch_size if selected else 256)
    if max_tokens < max_sequences:
        raise ValueError("resolved collection max_num_tokens must be at least max_batch_size")
    launch = normalize_probe_launch(
        {
            "identity": identity.model_dump(mode="json", exclude_none=True),
            "topology": topology,
            "precision": precision,
            "collection": {
                "max_model_len": search.context_length,
                "max_num_batched_tokens": max_tokens,
                "max_num_seqs": max_sequences,
                "gpu_memory_utilization": collection.memory_fraction,
                "prefill_cudagraph_policy": collection.prefill_cudagraph_policy,
                "max_prefill_cudagraph_size": collection.max_prefill_cudagraph_size,
            },
            "model_config": {
                "path": str(path),
                "sha256": model_config.sha256,
                **({"source_files": source_files} if source_files else {}),
            },
            "deployment": deployment,
        }
    )
    return payload, launch, resource_overrides


def _snapshot(index: Path, name: str, destination: Path) -> tuple[Path, list[str]]:
    """Copy one active attempt, verifying bytes without executing observer code."""
    from collector.fpm_forward.runtime_instrumentation import contained_file, load_instrumentation, read_json

    content = index.read_bytes()
    document = read_json(index)
    saved = deepcopy(_object(document["configurations"].get(name, {}), "observation configuration"))
    attempts = saved.get("attempts", [])
    if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
        raise ValueError("observation attempts must be a list of objects")
    active = [attempt for attempt in attempts if attempt.get("attempt_id") == saved.get("active_attempt_id")]
    if len(active) == 1:
        saved["attempts"] = active
    destination.mkdir(parents=True)
    diagnostics = []

    def copy_file(relative: str, digest: str | None = None) -> None:
        try:
            source = contained_file(index.parent, relative)
            data = source.read_bytes()
            if digest is not None and hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"artifact hash mismatch: {relative}")
            target = destination / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"conflicting snapshot artifact: {relative}")
            target.write_bytes(data)
        except (OSError, ValueError) as exc:
            diagnostics.append(str(exc))

    for attempt in saved.get("attempts", []):
        try:
            manifest_path = contained_file(index.parent, attempt["bundle"]["manifest"])
            bundle = load_instrumentation(manifest_path)
            if bundle.sha256 != attempt["bundle"]["sha256"]:
                raise ValueError("instrumentation bundle hash mismatch")
            copy_file(str(manifest_path.relative_to(index.parent)))
            for relative, digest in bundle.files.items():
                copy_file(str((bundle.root / relative).relative_to(index.parent)), digest)
            attempt["bundle"]["manifest"] = "raw/" + attempt["bundle"]["manifest"]
        except (KeyError, TypeError, OSError, ValueError) as exc:
            diagnostics.append(str(exc))
        for phase in _object(attempt.get("phases", {}), "observation phases").values():
            phase = _object(phase, "observation phase")
            for ref in [phase.get("launch_manifest", {}), *phase.get("artifacts", [])]:
                try:
                    copy_file(ref["path"], ref["sha256"])
                    ref["path"] = "raw/" + ref["path"]
                except (KeyError, TypeError) as exc:
                    diagnostics.append(f"invalid artifact reference: {exc}")
    (destination / "source-index.json").write_bytes(content)
    target = destination / "observations.json"
    target.write_text(_json({"schema_version": document.get("schema_version"), "configurations": {name: saved}}))
    return target, diagnostics


def runtime_probe_manifest(request: SupportRequest) -> dict[str, Any] | None:
    selected = request.profile_deployment()
    runtime = selected.resources.runtime_memory if selected else None
    if runtime is None:
        return None
    return _probe_provenance(runtime.provenance)


def _probe_provenance(provenance: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(provenance)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != PROVENANCE_SCHEMA:
        return None
    if (
        any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("checkpoint", "configuration", "observations_index")
        )
        or any(
            not isinstance(value.get(key), dict) for key in ("observed_resources", "model_metadata", "user_overrides")
        )
        or not isinstance(value.get("source_artifacts"), list)
        or any(
            not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str)
            for ref in value.get("source_artifacts", [])
        )
    ):
        raise ValueError("invalid onboarding runtime probe provenance")
    return value


def _resource_values(resources: dict[str, Any]) -> dict[str, Any]:
    values = deepcopy(resources)
    values.pop("provenance", None)
    values.get("runtime_memory", {}).pop("provenance", None)
    return values


def _request_launch(request: SupportRequest, launch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(launch)
    result["identity"] = request.identity.model_dump(mode="json", exclude_none=True)
    parallel = request.parallelism()
    result["topology"].update(
        tp=parallel["tensor"],
        dp=parallel["attention_data"],
        moe_tp=parallel["moe_tensor"],
        moe_ep=parallel["moe_expert"],
    )
    selected = request.profile_deployment()
    result["precision"] = {
        ("kvcache_quant_mode" if key == "kv_cache_dtype" else key): getattr(selected, key) for key in _PRECISION
    }
    scheduler = request.scheduler_limits()
    result["collection"].update(
        max_model_len=request.search.context_length,
        max_num_batched_tokens=scheduler["max_batched_tokens"],
        max_num_seqs=scheduler["max_sequences"],
        gpu_memory_utilization=request.collection.memory_fraction,
        prefill_cudagraph_policy=request.collection.prefill_cudagraph_policy,
        max_prefill_cudagraph_size=request.collection.max_prefill_cudagraph_size
        or (2048 if request.collection.prefill_cudagraph_policy == "explicit" else None),
    )
    return result


def verify_runtime_profile(request: SupportRequest) -> dict[str, Any] | None:
    """Revalidate source files and the exact runtime binding of an imported draft."""
    manifest = runtime_probe_manifest(request)
    if manifest is None:
        from .finalization import _merge_resources, _verify_collection, finalization_manifest
        from .plan import request_id

        finalized = finalization_manifest(request)
        if finalized is not None and "runtime_probe" in finalized:
            root = Path(finalized["source_directory"])
            original = SupportRequest.from_yaml(root / "request.yaml")
            if request_id(original) != finalized.get("source_request_id"):
                raise ValueError("finalized runtime source request changed")
            source = verify_runtime_profile(original)
            if source is None or finalized["runtime_probe"] != runtime_probe_manifest(original):
                raise ValueError("finalized runtime probe provenance changed")
            observations, evidence, _, _ = _verify_collection(
                original, root, memory_revision=finalized.get("memory_revision")
            )
            expected = _merge_resources(
                observations,
                evidence,
                source_references=finalized.get("observation_provenance") == "source_references",
            )
            if request.profile_deployment().resources.model_dump(mode="json") != expected:
                raise ValueError("finalized runtime resources differ from verified formal observations")
            launch = source["provenance"]["launch"]
            if _request_launch(request, launch) != launch:
                raise ValueError("finalized runtime settings changed; a compatible fresh probe is required")
            metadata = {
                key: getattr(request.fpm_profile, key) for key in ("architecture", "context_length", "num_experts")
            }
            if metadata != finalized["runtime_probe"]["model_metadata"]:
                raise ValueError("finalized model metadata changed; a compatible fresh probe is required")
        return None
    from collector.fpm_forward.runtime_instrumentation import read_json
    from collector.fpm_forward.runtime_observations import validate_observations

    covered = set()
    for ref in manifest["source_artifacts"]:
        path = Path(ref["path"])
        if path.is_symlink() or not path.is_file() or _hash(path) != ref["sha256"]:
            raise ValueError(f"runtime probe source artifact changed: {path}")
        covered.add(str(path))
    index = Path(manifest["observations_index"])
    if {str(index), str(index.parent / "launch.json"), str(index.parent / "model-config.json")} - covered:
        raise ValueError("runtime probe provenance is missing immutable source artifacts")
    name = manifest["configuration"]
    launch = read_json(index)["configurations"][name]["launch"]
    result = validate_observations(index, {name: launch})[name]
    if result["status"] != "complete":
        raise ValueError("runtime probe evidence is incomplete: " + "; ".join(result["diagnostics"]))
    if _request_launch(request, launch) != launch:
        raise ValueError("runtime-affecting draft edits require a compatible fresh probe")
    expected = _resource_values(result["resources"])
    if manifest["observed_resources"] != expected:
        raise ValueError("saved observed resources differ from immutable runtime evidence")
    selected = request.profile_deployment()
    actual = _resource_values(selected.resources.model_dump(mode="json"))
    observed_capacity = expected["runtime_memory"].pop("kv_cache_bytes")
    actual_capacity = actual["runtime_memory"].pop("kv_cache_bytes")
    if actual != expected:
        raise ValueError("runtime cache geometry or settings changed; a compatible fresh probe is required")
    overrides = (
        {}
        if actual_capacity == observed_capacity
        else {
            "runtime_memory.kv_cache_bytes": {
                "observed": observed_capacity,
                "value": actual_capacity,
                "source": "user override; assumption, not a runtime measurement",
            }
        }
    )
    if manifest.get("user_overrides", {}) != overrides:
        raise ValueError(
            "resource edits require checkpoint registration to preserve observed values and label user overrides"
        )
    metadata = {key: getattr(request.fpm_profile, key) for key in ("architecture", "context_length", "num_experts")}
    if metadata != manifest["model_metadata"]:
        raise ValueError("model metadata edits require a compatible fresh probe")
    return result


def preserve_runtime_overrides(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Keep observed source values when checkpoint edits change a resource assumption."""
    profile = previous.get("fpm_profile")
    deployments = profile.get("deployments", []) if isinstance(profile, dict) else []
    if not isinstance(deployments, list):
        return current
    payload = deepcopy(current)
    for position, deployment in enumerate(deployments):
        if not isinstance(deployment, dict):
            continue
        resources = deployment.get("resources")
        runtime = resources.get("runtime_memory") if isinstance(resources, dict) else None
        manifest = _probe_provenance(runtime.get("provenance")) if isinstance(runtime, dict) else None
        if manifest is None or not payload.get("fpm_profile"):
            continue
        # Removing the whole profile leaves an incomplete draft. Its prior
        # request/evidence and acceptance are retained by checkpoint history.
        updated_deployments = payload["fpm_profile"].get("deployments", [])
        if position >= len(updated_deployments):
            continue
        updated_resources = updated_deployments[position].get("resources", {})
        updated_runtime = updated_resources.get("runtime_memory", {})
        updated = _probe_provenance(updated_runtime.get("provenance")) if isinstance(updated_runtime, dict) else None
        if updated is None:
            from .finalization import finalization_manifest
            from .plan import request_id

            request = SupportRequest.model_validate(current)
            finalized = finalization_manifest(request)
            if finalized is not None and (
                (
                    finalized.get("runtime_probe") == manifest
                    and finalized.get("source_request_id") == request_id(SupportRequest.model_validate(previous))
                )
                or (
                    isinstance(finalized.get("memory_revision"), dict)
                    and finalized["memory_revision"].get("request_id")
                    == request_id(SupportRequest.model_validate(previous))
                )
            ):
                verify_runtime_profile(request)
                return current
            raise ValueError(
                "runtime evidence cannot be removed from an imported profile; import a fresh compatible probe"
            )
        if updated["observations_index"] != manifest["observations_index"]:
            verify_runtime_profile(SupportRequest.model_validate(current))
            return current
        # Do not validate a complete simulation request here: runtime settings
        # may be edited before a compatible replacement probe is available.
        capacity = updated_runtime.get("kv_cache_bytes")
        observed = manifest["observed_resources"]["runtime_memory"]["kv_cache_bytes"]
        manifest["user_overrides"] = (
            {}
            if capacity == observed
            else {
                "runtime_memory.kv_cache_bytes": {
                    "observed": observed,
                    "value": capacity,
                    "source": "user override; assumption, not a runtime measurement",
                }
            }
        )
        updated_runtime["provenance"] = _json(manifest)
    return payload


def verify_checkpoint_runtime(state: Any, name: str, checkpoint: Path) -> None:
    request = SupportRequest.model_validate(state.configurations[name].draft_request)
    manifest = runtime_probe_manifest(request)
    if manifest is None:
        return
    from collector.fpm_forward.runtime_instrumentation import read_json

    _, launch, _ = checkpoint_launch(state, name, checkpoint)
    saved = read_json(Path(manifest["observations_index"]))["configurations"][name]["launch"]
    if launch != saved:
        raise ValueError("checkpoint runtime inputs changed; a compatible fresh probe is required")


def runtime_collection_inputs(
    request: SupportRequest, deployment: FPMDeployment | None
) -> tuple[list[str], FPMDeployment | None]:
    """Supply the collector's explicit opt-in flags from verified source evidence."""
    result = verify_runtime_profile(request)
    if result is None:
        return [], deployment
    manifest = runtime_probe_manifest(request)
    root = Path(manifest["observations_index"]).parent
    observed_deployment = FPMDeployment.model_validate(result["provenance"]["launch"]["deployment"])
    if deployment is not None and deployment != observed_deployment:
        raise ValueError(
            "collection deployment differs from the runtime probe; use matching options or probe the changed deployment"
        )
    return [
        "--fpm-runtime-instrumentation",
        str(root / result["provenance"]["instrumentation"]["manifest"]),
        "--fpm-runtime-launch",
        str(root / "launch.json"),
        "--fpm-runtime-configuration",
        manifest["configuration"],
    ], observed_deployment


def verify_runtime_acceptance(request: SupportRequest) -> None:
    manifest = runtime_probe_manifest(request)
    if manifest is None:
        return
    from .checkpoint import _canonical_request, _load, report

    checkpoint = Path(manifest["checkpoint"])
    state = _load(checkpoint)
    name = manifest["configuration"]
    if name not in state.configurations or not report(state, checkpoint)["configurations"][name]["profile_accepted"]:
        raise ValueError(
            "formal collection requires explicit acceptance of the imported runtime profile in its checkpoint"
        )
    if _canonical_request(state.configurations[name], require_complete=True) != request.model_dump(
        mode="json", exclude_none=True
    ):
        raise ValueError("formal collection request differs from the accepted checkpoint draft")


def verify_collection_runtime(
    request: SupportRequest,
    observations: Path,
    *,
    collection_checkpoint: dict[str, Any] | None = None,
    memory_request: SupportRequest | None = None,
) -> dict[str, Any]:
    """Validate new raw observations before asserting conservative compatibility."""
    from collector.fpm_forward.runtime_observations import validate_observations

    source = verify_runtime_profile(request)
    if source is None:
        raise ValueError("formal runtime comparison requires imported probe evidence")
    manifest = runtime_probe_manifest(request)
    name = manifest["configuration"]
    result = validate_observations(observations, {name: source["provenance"]["launch"]})[name]
    if result["status"] != "complete":
        raise ValueError("formal runtime observations are incomplete: " + "; ".join(result["diagnostics"]))
    if collection_checkpoint is not None:
        from collector.fpm_forward.runtime_instrumentation import contained_file, read_json

        entry = read_json(observations)["configurations"][name]
        active = entry["active_attempt_id"]
        if collection_checkpoint.get("runtime_observation_attempt_id") != active:
            raise ValueError("formal runtime observations belong to a different collection attempt")
        attempt = next(value for value in entry["attempts"] if value["attempt_id"] == active)
        seen = set()
        for phase in attempt["phases"].values():
            context = read_json(contained_file(observations.parent, phase["launch_manifest"]["path"]))
            cell = context.get("cell_id")
            native_attempt = context.get("collector_attempt_id")
            saved = collection_checkpoint.get("cells", {}).get(cell, {})
            if (
                not isinstance(cell, str)
                or cell in seen
                or not isinstance(native_attempt, str)
                or phase.get("cell_id") != cell
                or phase.get("collector_attempt_id") != native_attempt
                or saved.get("status") != "passed"
                or saved.get("attempt_id") != native_attempt
            ):
                raise ValueError("formal runtime phase does not match the successful native cell attempt")
            seen.add(cell)
        if seen != set(collection_checkpoint.get("cells", {})):
            raise ValueError("formal runtime observations do not cover every collected cell")
    for field in ("bundle_sha256", "runtime", "runtime_settings"):
        if source["provenance"][field] != result["provenance"][field]:
            raise ValueError(f"formal runtime {field} differs from the accepted probe; review a fresh compatible probe")

    def geometry(evidence):
        groups = next(
            item["evidence"]["cache"]["groups"]
            for item in evidence["artifacts"]
            if item["evidence"]["kind"] == "worker"
        )
        return [{key: value for key, value in group.items() if key != "pool_id"} for group in groups]

    if geometry(source["provenance"]) != geometry(result["provenance"]):
        raise ValueError("formal runtime cache group geometry differs from the accepted probe")
    expected = _resource_values(source["resources"])
    actual = _resource_values(result["resources"])
    expected["runtime_memory"].pop("kv_cache_bytes")
    capacity = actual["runtime_memory"].pop("kv_cache_bytes")
    if actual != expected:
        raise ValueError("formal runtime cache layout or settings differ from the accepted probe")
    bound = request.profile_deployment().resources.runtime_memory.kv_cache_bytes
    if memory_request is not None:
        if collection_checkpoint is None:
            raise ValueError("memory revision requires complete formal collection attempt bindings")
        _verify_memory_revision(request, memory_request, result)
        bound = memory_request.profile_deployment().resources.runtime_memory.kv_cache_bytes
    if capacity < bound:
        raise ValueError(
            f"formal runtime usable capacity {capacity} is smaller than the accepted bound {bound}; review new evidence"
        )
    result["compatibility"] = {
        "status": "compatible",
        "accepted_capacity_bytes": bound,
        "observed_capacity_bytes": capacity,
        "policy": "unchanged runtime/settings/geometry and observed usable capacity at least the accepted bound",
        "source_observations_index": manifest["observations_index"],
        "observations_index": str(observations),
        "observations_sha256": _hash(observations),
    }
    return result


def _verify_memory_revision(original: SupportRequest, revised: SupportRequest, observed: dict[str, Any]) -> None:
    """Allow only a smaller capacity measured by these exact formal attempts."""
    selected = original.profile_deployment()
    position = original.fpm_profile.deployments.index(selected)
    payload = revised.model_dump(mode="json")
    try:
        resources = payload["fpm_profile"]["deployments"][position]["resources"]
        capacity = resources["runtime_memory"]["kv_cache_bytes"]
        resources["runtime_memory"]["kv_cache_bytes"] = selected.resources.runtime_memory.kv_cache_bytes
        resources["runtime_memory"]["provenance"] = selected.resources.runtime_memory.provenance
        resources["provenance"] = selected.resources.provenance
        payload["fpm_profile"]["provenance"] = original.fpm_profile.provenance
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("memory revision requires the original runtime profile and deployment") from exc
    if payload != original.model_dump(mode="json"):
        raise ValueError("memory revision changed non-capacity onboarding inputs")
    if capacity >= selected.resources.runtime_memory.kv_cache_bytes:
        raise ValueError("memory revision must lower the original accepted capacity")
    source = runtime_probe_manifest(original)
    revision = runtime_probe_manifest(revised)
    if (
        source is None
        or revision is None
        or any(revision[key] != source[key] for key in ("checkpoint", "configuration", "model_metadata"))
        or revision["user_overrides"]
    ):
        raise ValueError("memory revision requires imported formal observations without capacity overrides")
    verified = verify_runtime_profile(revised)
    if verified is None or capacity != observed["resources"]["runtime_memory"]["kv_cache_bytes"]:
        raise ValueError("memory revision capacity must equal the verified formal observed minimum")

    def identity(evidence):
        value = deepcopy(evidence)
        value.pop("observations_index", None)
        # Observation imports relocate immutable bytes into a snapshot. Paths
        # can change; attempts, record contents and every artifact digest cannot.
        for field in ("artifacts", "launch_artifacts", "runtime_artifacts"):
            value[field] = sorted(
                [{key: item for key, item in ref.items() if key != "path"} for ref in value[field]],
                key=_json,
            )
        value["instrumentation"].pop("manifest")
        return value

    if identity(verified["provenance"]) != identity(observed["provenance"]):
        raise ValueError("memory revision is not derived from the same complete formal collection observations")


def _new_root(value: str, checkpoint: Path, *, fresh: bool) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ValueError("runtime output must not be a symlink")
    root = raw.resolve()
    if checkpoint == root or checkpoint.is_relative_to(root):
        raise ValueError("keep the session checkpoint outside runtime output directories")
    if fresh and root.exists():
        raise ValueError("runtime import output must be a fresh directory")
    return root


def _selected(state: Any, requested: list[str] | None) -> list[str]:
    names = requested or list(state.configurations)
    if not names or len(names) != len(set(names)):
        raise ValueError("select at least one unique checkpoint configuration")
    unknown = set(names) - state.configurations.keys()
    if unknown:
        raise ValueError("unknown checkpoint configurations: " + ", ".join(sorted(unknown)))
    return names


def _draft(
    payload: dict[str, Any],
    launch: dict[str, Any],
    overrides: dict[str, Any],
    result: dict[str, Any],
    snapshot: Path,
    checkpoint: Path,
    name: str,
) -> SupportRequest:
    # Full workload validation belongs to simulation draft construction. A
    # prior profile may describe different settings; replace it only after
    # independently validating the newly observed resources below.
    request = SupportRequest.model_validate({**payload, "fpm_profile": None})
    precision = {
        ("kv_cache_dtype" if key == "kvcache_quant_mode" else key): value for key, value in launch["precision"].items()
    }
    model = load_model_config(launch["model_config"]["path"])
    if model.sha256 != launch["model_config"]["sha256"]:
        raise ValueError("model config changed during runtime import")
    profile = profile_from_observations(model, request, precision, result["resources"], overrides)
    if payload.get("fpm_profile"):
        provenance = json.loads(profile.provenance)
        provenance["source_profile_provenance"] = payload["fpm_profile"].get("provenance")
        profile.provenance = _json(provenance)
    manifest = {
        "schema_version": PROVENANCE_SCHEMA,
        "checkpoint": str(checkpoint),
        "configuration": name,
        "observations_index": str(snapshot),
        "source_artifacts": [
            {"path": str(path), "sha256": _hash(path)} for path in sorted(snapshot.parent.rglob("*")) if path.is_file()
        ],
        "observed_resources": _resource_values(result["resources"]),
        "model_metadata": {key: getattr(profile, key) for key in ("architecture", "context_length", "num_experts")},
        "user_overrides": {},
        "review_status": "requires_review",
        "timing_status": "not_established_by_runtime_probe",
    }
    profile.deployments[0].resources.runtime_memory.provenance = _json(manifest)
    payload = deepcopy(payload)
    payload["fpm_profile"] = profile.model_dump(mode="json", exclude_none=True)
    return SupportRequest.model_validate(payload)


def import_observations(
    checkpoint: Path, observations: Path, output: Path, configurations: list[str] | None = None
) -> dict[str, Any]:
    from collector.fpm_forward.runtime_instrumentation import read_json
    from collector.fpm_forward.runtime_observations import validate_observations

    from .checkpoint import _load, save_checkpoint

    state = _load(checkpoint)
    names = _selected(state, configurations)
    output = _new_root(str(output), checkpoint, fresh=True)
    if observations.is_symlink():
        raise ValueError("runtime observation index cannot be a symlink")
    document = read_json(observations)
    if document.get("schema_version") != "aisimulate-runtime-observations/v1" or not isinstance(
        document.get("configurations"), dict
    ):
        raise ValueError("unsupported runtime observation index schema")
    output.mkdir(parents=True)
    results, patches = {}, {}
    for ordinal, name in enumerate(names, 1):
        directory = output / f"configuration-{ordinal:03d}"
        directory.mkdir()
        result = {
            "status": "incomplete",
            "resources": None,
            "diagnostics": [],
            "provenance": {"source_index": str(observations)},
        }
        request = None
        try:
            snapshot, diagnostics = _snapshot(observations, name, directory / "evidence")
            original, launch, overrides = checkpoint_launch(state, name, checkpoint)
            validated = validate_observations(snapshot, {name: launch})[name]
            result.update(validated)
            result["diagnostics"] += diagnostics
            if diagnostics:
                result["status"] = "incomplete"
            if result["status"] == "complete":
                # Frozen formal inputs belong to this configuration's immutable
                # snapshot, alongside its model configuration and raw records.
                (snapshot.parent / "launch.json").write_text(_json(launch))
                (snapshot.parent / "model-config.json").write_bytes(Path(launch["model_config"]["path"]).read_bytes())
                for relative, digest in launch["model_config"].get("source_files", {}).items():
                    source = Path(launch["model_config"]["path"]).parent / relative
                    content = source.read_bytes()
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError(f"model configuration sidecar changed during import: {relative}")
                    (snapshot.parent / relative).write_bytes(content)
                request = _draft(original, launch, overrides, result, snapshot, checkpoint, name)
                (directory / "request.yaml").write_text(
                    yaml.safe_dump(request.model_dump(mode="json", exclude_none=True), sort_keys=False)
                )
                (directory / "fpm-model-profile.json").write_text(
                    _json(request.fpm_profile.model_dump(mode="json", exclude_none=True))
                )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["status"] = "incomplete"
            result["diagnostics"].append(str(exc))
            (directory / "source-index.json").write_bytes(observations.read_bytes())
        validation = directory / "validation.json"
        validation.write_text(_json(result))
        old = state.configurations[name]
        artifacts = {}
        if request is not None and result["status"] == "complete":
            artifacts.update(
                {
                    key: {"archived": True}
                    for key, ref in old.artifacts.items()
                    if not ref.archived
                    and (ref.kind in {"request", "profile"} or ref.scope != "input" or key.startswith("runtime-"))
                }
            )
        token = uuid.uuid4().hex[:12]
        for number, path in enumerate(sorted(directory.rglob("*"))):
            if path.is_file():
                kind = (
                    "request"
                    if path.name == "request.yaml"
                    else "profile"
                    if path.name == "fpm-model-profile.json"
                    else "file"
                )
                artifacts[f"runtime-{token}-{number}"] = {
                    "path": str(path),
                    "kind": kind,
                    "scope": "input",
                    "archived": result["status"] != "complete",
                }
        patch = {
            "artifacts": artifacts,
            "history": [
                *old.history,
                {
                    "event": "runtime_observations_import",
                    "validation": str(validation),
                    "sha256": _hash(validation),
                    "status": result["status"],
                    "diagnostics": result["diagnostics"],
                    "archived_artifacts": [key for key, ref in artifacts.items() if ref.get("archived")],
                    "reason": "Fresh evidence supersedes prior drafts and dependent collection/validation outputs."
                    if result["status"] == "complete"
                    else "Failed import retained as historical evidence; current profile inputs are unchanged.",
                },
            ],
            "progress": {
                "stage": 3,
                "status": "in_progress" if result["status"] == "complete" else "blocked",
                "blockers": result["diagnostics"],
                "next_action": "Review and accept this exact draft."
                if result["status"] == "complete"
                else "Resolve runtime evidence diagnostics and import a fresh attempt.",
            },
        }
        if request is not None and result["status"] == "complete":
            patch["draft_request"] = request.model_dump(mode="json", exclude_none=True)
        patches[name] = patch
        results[name] = {
            "status": result["status"],
            "diagnostics": result["diagnostics"],
            "validation": str(validation),
        }
        if request is not None and result["status"] == "complete":
            results[name].update(
                request=str(directory / "request.yaml"), profile=str(directory / "fpm-model-profile.json")
            )
    state, _ = save_checkpoint(
        checkpoint, patch={"configurations": patches}, expected_revision=state.revision, accept=[]
    )
    result = {
        "checkpoint": str(checkpoint),
        "revision": state.revision,
        "status": "complete" if all(value["status"] == "complete" for value in results.values()) else "incomplete",
        "configurations": results,
    }
    (output / "import.json").write_text(_json(result))
    return result


def run_runtime_command(args: argparse.Namespace) -> int:
    from collector.fpm_forward.runtime_probe import probe_runtime

    from .checkpoint import _load, _target, save_checkpoint

    checkpoint = _target(args.checkpoint)
    if args.support_action == "import-observations":
        result = import_observations(
            checkpoint, Path(args.observations).expanduser().absolute(), Path(args.output_dir), args.configuration
        )
        print(_json(result), end="")
        return 0 if result["status"] == "complete" else 1
    state = _load(checkpoint)
    names = _selected(state, args.configuration)
    output = _new_root(args.output_dir, checkpoint, fresh=False)
    overrides = {key: getattr(args, key) for key in FPMDeployment.model_fields if getattr(args, key, None) is not None}
    launches, failed = {}, {}
    for name in names:
        try:
            _, launches[name], _ = checkpoint_launch(
                state,
                name,
                checkpoint,
                overrides,
                resolve_cpu_defaults=not args.resume,
                resume_output_dir=output if args.resume else None,
            )
        except (ValueError, OSError) as exc:
            failed[name] = {"status": "incomplete", "diagnostics": [str(exc)]}
    result = (
        probe_runtime(
            launches, instrumentation=args.instrumentation, output_dir=output, execute=args.execute, resume=args.resume
        )
        if launches
        else {"status": "failed", "configurations": {}}
    )
    result["configurations"].update(failed)
    if failed:
        result["status"] = "partial" if launches else "failed"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"onboarding-probe-{uuid.uuid4().hex}.json"
    path.write_text(_json(result))
    patches = {}
    for name in names:
        config = state.configurations[name]
        patch = {
            "history": [
                *config.history,
                {
                    "event": "runtime_probe",
                    "report": str(path),
                    "sha256": _hash(path),
                    "result": result["configurations"][name],
                },
            ]
        }
        if name in launches:
            patch["inputs"] = {"collection_deployment": launches[name]["deployment"]}
        patches[name] = patch
    state, _ = save_checkpoint(
        checkpoint, patch={"configurations": patches}, expected_revision=state.revision, accept=[]
    )
    result.update(checkpoint=str(checkpoint), revision=state.revision)
    print(_json(result), end="")
    return 0 if result["status"] in {"preview", "completed"} else 1
