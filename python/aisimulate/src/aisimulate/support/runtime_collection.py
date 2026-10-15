# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit frozen-plan bindings for independently collected memory evidence."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from aisimulate.fpm_contract import FPM_FINALIZATION_TIMEOUT_ENV

from .schema import SupportRequest


def _file(path: str | Path, *, model_config: bool = False) -> dict[str, str]:
    path = Path(path).absolute()
    if (path.is_symlink() and not model_config) or not path.is_file():
        raise ValueError(f"runtime evidence must be a regular file: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(reference: dict[str, str], *, plan: bool = False) -> tuple[Path, dict[str, Any]]:
    from collector.fpm_forward.runtime_instrumentation import read_json, validate_sha256
    from collector.fpm_forward.runtime_memory import validate_saved_plan

    path = Path(reference["path"]).absolute()
    actual = _file(path)
    validate_sha256(reference["sha256"], "bound collection plan" if plan else "bound runtime evidence")
    document = read_json(path)
    if plan:
        validate_saved_plan(document)
    if (document["sha256"] if plan else actual["sha256"]) != reference["sha256"]:
        raise ValueError(f"bound {'plan' if plan else 'artifact'} hash mismatch: {path}")
    return path, document


def checkpoint_observation_index(checkpoint_path: Path, checkpoint: dict[str, Any]) -> Path:
    """Resolve a checkpoint's recorded observation index to a regular file.

    Relative paths are anchored to the checkpoint directory. Reject a symlink
    at the index itself; callers separately verify containment and byte hashes.
    """
    recorded = checkpoint.get("runtime_observations")
    if not isinstance(recorded, str) or not recorded.strip():
        raise ValueError("collector checkpoint records no runtime observation index")
    index = Path(recorded)
    if not index.is_absolute():
        index = checkpoint_path.resolve().parent / index
    if index.is_symlink() or not index.is_file():
        raise ValueError(f"runtime observation index must be a regular file: {index}")
    return index.resolve()


def _differences(accepted: Any, collection: Any, field: str = "") -> list[dict[str, Any]]:
    if accepted == collection:
        return []
    if isinstance(accepted, dict) and isinstance(collection, dict):
        return [
            difference
            for key in sorted(accepted.keys() | collection.keys())
            for difference in _differences(
                accepted.get(key, {"missing": True}), collection.get(key, {"missing": True}), field + "/" + key
            )
        ]
    return [{"field": field, "accepted": accepted, "collection": collection}]


def _source(request: SupportRequest) -> dict[str, Any]:
    from .runtime import verify_runtime_profile

    source = verify_runtime_profile(request)
    if source is None:
        raise ValueError("bound runtime comparison requires imported probe evidence")
    return source


def _collection_controls(cells, deployment):
    """Separate recorded collection controls for memory-only review, without editing plans."""
    from collector.fpm_forward.planner import BackendPolicy

    from aisimulate.generator.builders.fpm_builder import _benchmark_wait_timeout_seconds

    compared_cells, changes = [], []
    for cell in cells:
        policy = cell.backend_policy.to_dict()
        overrides = policy["generator_overrides"]
        params = overrides.get("params", {})
        agg = params.get("agg", {})
        args = agg.get("extra_cli_args", [])
        if not isinstance(args, list) or any(not isinstance(token, str) for token in args):
            raise ValueError("collection policy extra_cli_args must be a list of strings")
        remaining, deadline = [], []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--benchmark-timeout" or token.startswith("--benchmark-timeout="):
                if deadline:
                    raise ValueError("collection policy contains duplicate --benchmark-timeout controls")
                width = 2 if token == "--benchmark-timeout" else 1
                deadline = args[index : index + width]
                _benchmark_wait_timeout_seconds(deadline)
                index += width
            else:
                remaining.append(token)
                index += 1
        if "extra_cli_args" in agg:
            agg["extra_cli_args"] = remaining
            if not remaining:
                del agg["extra_cli_args"]
                if not agg:
                    params.pop("agg")
                if not params:
                    overrides.pop("params")
        compared_cells.append(replace(cell, backend_policy=BackendPolicy(**policy)))
        changes.extend(_differences([], deadline, f"/collection_controls/{cell.workload_kind}/benchmark_timeout"))

    deployment = deepcopy(deployment)
    k8s = deployment.get("K8sConfig", {})
    environment = k8s.get("extra_env", [])
    known = {
        "DYN_BENCH_CONTENT_SEED",
        "DYN_BENCH_PREFILL_CONTENT",
        "DYN_BENCH_POOL_TAG",
        "DYN_BENCH_GIANT_KV_THRESHOLD",
        "DYN_BENCH_GIANT_KV_REPEATS",
        "DYN_FORWARDPASS_METRIC_PORT",
        "FPM_RESULT_WAIT_TIMEOUT_SECONDS",
        FPM_FINALIZATION_TIMEOUT_ENV,
    }
    retained, seen = [], set()
    if not isinstance(environment, list):
        raise ValueError("collection environment must be a list")
    for item in environment:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("collection environment requires named entries")
        name = item["name"]
        if name not in known:
            retained.append(item)
            continue
        value = item.get("value")
        if (
            name in seen
            or set(item) != {"name", "value"}
            or not isinstance(value, str)
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError(f"collection control {name} requires one literal string value")
        seen.add(name)
        if name == "DYN_BENCH_PREFILL_CONTENT" and value not in {"random", "sharegpt", "sharegpt_chain", "zeros"}:
            raise ValueError(f"collection control {name} has an unsupported content mode")
        if name in {"DYN_BENCH_GIANT_KV_THRESHOLD", "DYN_BENCH_GIANT_KV_REPEATS"} and not re.fullmatch(
            r"0|[1-9][0-9]*", value
        ):
            raise ValueError(f"collection control {name} must be a nonnegative integer")
        if name in {"DYN_FORWARDPASS_METRIC_PORT", "FPM_RESULT_WAIT_TIMEOUT_SECONDS", FPM_FINALIZATION_TIMEOUT_ENV}:
            maximum = {
                "DYN_FORWARDPASS_METRIC_PORT": 65535,
                "FPM_RESULT_WAIT_TIMEOUT_SECONDS": 43200,
                FPM_FINALIZATION_TIMEOUT_ENV: 3600,
            }[name]
            if not re.fullmatch(r"[1-9][0-9]*", value) or len(value) > len(str(maximum)) or int(value) > maximum:
                raise ValueError(f"collection control {name} must be an integer from 1 through {maximum}")
        changes.extend(_differences({"missing": True}, value, "/collection_controls/environment/" + name))
    if "extra_env" in k8s:
        k8s["extra_env"] = retained
    return tuple(compared_cells), deployment, changes


def _plan(request: SupportRequest, binding: dict[str, Any], source: dict[str, Any]):
    from collector.fpm_forward.repeatability import load_repeatability_deployment, load_repeatability_source
    from collector.fpm_forward.runtime_probe import validate_collection_probe_launch

    path, saved = _read(binding["plan"], plan=True)
    if path.name != "collection-plan.json":
        raise ValueError("bound runtime plan must be the archived collection-plan.json")
    plan = load_repeatability_source(path.parent)
    reference = saved.get("runtime_observation")
    if (
        not isinstance(reference, dict)
        or not isinstance(reference.get("configuration"), str)
        or not reference["configuration"].strip()
    ):
        raise ValueError("bound collection plan lacks a recorded runtime configuration")
    launch = plan.runtime_launch
    original = source["provenance"]["launch"]
    # Paths locate config bytes; their hashes, sidecars and model revision identify
    # the checkpoint. All serving, precision and scheduler facts remain exact.
    expected, actual = deepcopy(original), deepcopy(launch)
    for value in (expected, actual):
        value.pop("deployment")
        # AISimulate source provenance is reviewed below, separately from the checkpoint.
        value["identity"].pop("aisimulate_revision", None)
        value["model_config"].pop("path")
        value["model_config"].setdefault("source_files", {})
    if actual != expected:
        raise ValueError(
            "collection immutable model/precision/settings differ: " + json.dumps(_differences(expected, actual))
        )
    phase = binding["phase"]
    cells = [cell for cell in plan.cells if cell.workload_kind == phase]
    if phase not in request.collection_phases or len(cells) != 1:
        raise ValueError("bound phase requires exactly one matching frozen collection cell")
    bundle = plan.runtime_instrumentation
    deployment = load_repeatability_deployment(path.parent)
    compared_cells, compared_deployment, controls = _collection_controls(plan.cells, deployment)
    validate_collection_probe_launch(
        launch,
        bundle,
        model_path=plan.model_path,
        system=plan.system,
        backend=plan.backend,
        backend_version=plan.capability.aic_database_version,
        cells=compared_cells,
        options=plan.options,
        profile=plan.fpm_profile,
        generator_overrides=compared_deployment,
        phases=tuple(cell.workload_kind for cell in plan.cells),
    )
    reviewed = {
        "accepted_bundle_sha256": source["provenance"]["bundle_sha256"],
        "collection_bundle_sha256": bundle.sha256,
        "collector_revision": plan.aic_revision,
        "generator_config_sha256": plan.generator_config_sha256,
        "changes": _differences(
            {
                "bundle_sha256": source["provenance"]["bundle_sha256"],
                "runtime": source["provenance"]["runtime"],
                "deployment": original["deployment"],
                "identity": original["identity"],
            },
            {
                "bundle_sha256": bundle.sha256,
                "runtime": bundle.manifest["runtime"],
                "deployment": launch["deployment"],
                "identity": launch["identity"],
            },
        ),
    }
    reviewed["changes"] = sorted([*reviewed["changes"], *controls], key=lambda item: item["field"])
    return path, plan, cells[0], bundle, reviewed


def collection_runtime_review(request: SupportRequest, binding: dict[str, Any]) -> dict[str, Any]:
    """Describe exact runtime changes for review; never qualify observations.

    A caller must inspect these changes and add a recorded ``rationale`` before
    passing the resulting review in a binding. Settings mismatches still fail.
    """
    return _plan(request, binding, _source(request))[-1]


def _plans(request: SupportRequest, bindings: list[dict[str, Any]], source: dict[str, Any]):
    phases = [item["phase"] for item in bindings]
    if len(set(phases)) != len(phases) or set(phases) != set(request.collection_phases):
        raise ValueError("runtime bindings require each serving-role phase exactly once")
    verified = []
    for binding in bindings:
        try:
            values = _plan(request, binding, source)
            review = binding.get("review")
            required = values[-1]
            if (required["changes"] or review is not None) and (
                not isinstance(review, dict)
                or not isinstance(review.get("rationale"), str)
                or not review["rationale"].strip()
                or {key: value for key, value in review.items() if key != "rationale"} != required
            ):
                raise ValueError("exact runtime change review required: " + json.dumps(required, sort_keys=True))
            verified.append((binding, *values))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"{binding['phase']}: {error}") from error
    return verified


def verify_collection_runtime_plans(request: SupportRequest, bindings: list[dict[str, Any]]) -> dict[str, Any]:
    """Check all phase plans before attempts exist; this is not memory evidence."""
    verified = _plans(request, bindings, _source(request))
    return {
        "status": "verified",
        "scope": "frozen collection plans and reviewed runtime changes only; live evidence remains required",
        "phases": {
            binding["phase"]: {
                "plan": deepcopy(binding["plan"]),
                "configuration": plan.runtime_configuration,
                "review": deepcopy(binding.get("review")),
            }
            for binding, _, plan, _, _, _ in verified
        },
    }


def _worker(result):
    return next(item["evidence"] for item in result["provenance"]["artifacts"] if item["evidence"]["kind"] == "worker")


def _geometry(result):
    groups = _worker(result)["cache"]["groups"]
    return [{key: value for key, value in group.items() if key != "pool_id"} for group in groups]


def _evidence(index: Path, result: dict[str, Any]) -> list[dict[str, str]]:
    from collector.fpm_forward.runtime_instrumentation import contained_file

    provenance = result["provenance"]
    paths = [index]
    for field in ("artifacts", "launch_artifacts", "runtime_artifacts"):
        paths.extend(contained_file(index.parent, ref["path"]) for ref in provenance[field])
    instrumentation = provenance["instrumentation"]
    manifest = contained_file(index.parent, instrumentation["manifest"])
    paths.append(manifest)
    paths.extend(contained_file(manifest.parent, name) for name in instrumentation["files"])
    return [_file(path) for path in paths]


def verify_collection_runtime_bindings(request: SupportRequest, bindings: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify distinct successful phase attempts against unchanged accepted memory.

    Plan references carry the producer's semantic SHA; checkpoint/index references
    carry byte hashes. Recorded names, launch and bundle derive only from the plan.
    """
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA
    from collector.fpm_forward.runtime_instrumentation import contained_file, read_json
    from collector.fpm_forward.runtime_observations import validate_observation_phase

    from .runtime import _resource_values, runtime_probe_manifest

    source = _source(request)
    verified = _plans(request, bindings, source)
    manifest = runtime_probe_manifest(request)
    inputs = [*_evidence(Path(manifest["observations_index"]), source), *manifest["source_artifacts"]]
    phases = {}
    bound = request.profile_deployment().resources.runtime_memory.kv_cache_bytes
    for binding, plan_path, plan, cell, bundle, _ in verified:
        phase = binding["phase"]
        try:
            checkpoint_path, checkpoint = _read(binding["checkpoint"])
            index, document = _read(binding["observations"])
            if not index.resolve().is_relative_to(plan_path.parent.resolve()):
                raise ValueError("runtime observation index escapes the frozen collection directory")
            if (
                checkpoint.get("schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("plan_sha256") != plan.sha256
                or checkpoint_observation_index(checkpoint_path, checkpoint) != index.resolve()
                or checkpoint.get("runtime_observation_attempt_id") != binding["attempt_id"]
                or set(checkpoint.get("cells", {})) - {value.cell_id for value in plan.cells}
            ):
                raise ValueError("runtime checkpoint does not match the bound plan/index/attempt")
            saved = checkpoint.get("cells", {}).get(cell.cell_id, {})
            if (
                not isinstance(binding["collector_attempt_id"], str)
                or not binding["collector_attempt_id"].strip()
                or saved.get("status") != "passed"
                or saved.get("attempt_id") != binding["collector_attempt_id"]
                or saved.get("cleanup_error")
            ):
                raise ValueError("runtime phase does not match the successful native cell attempt")
            result = validate_observation_phase(
                index, plan.runtime_configuration, plan.runtime_launch, phase=phase, attempt_id=binding["attempt_id"]
            )
            if result["provenance"]["bundle_sha256"] != bundle.sha256:
                raise ValueError("runtime observation bundle differs from the frozen plan")
            entry = document["configurations"][plan.runtime_configuration]
            attempt = next(value for value in entry["attempts"] if value["attempt_id"] == binding["attempt_id"])
            recorded = attempt["phases"][phase]
            context = read_json(contained_file(index.parent, recorded["launch_manifest"]["path"]))
            for value in (recorded, context):
                if (
                    value.get("cell_id") != cell.cell_id
                    or value.get("collector_attempt_id") != binding["collector_attempt_id"]
                ):
                    raise ValueError("runtime phase does not match the successful native cell attempt")
            if recorded.get("status") != "passed":
                raise ValueError("runtime phase is not a passed collection attempt")
            original_settings, settings = (
                deepcopy(source["provenance"]["runtime_settings"]),
                deepcopy(result["provenance"]["runtime_settings"]),
            )
            settings_diff = _differences(original_settings, settings, "/runtime_settings")
            # Both spellings already passed _matches_model_path and revision/hash
            # checks independently. Do not normalize any other effective setting.
            settings["model_config"]["model"] = original_settings["model_config"]["model"]
            if settings != original_settings:
                raise ValueError(
                    "effective runtime/graph settings differ from accepted probe: " + json.dumps(settings_diff)
                )
            if _geometry(result) != _geometry(source):
                raise ValueError("runtime cache group geometry differs from the accepted probe")
            for field in ("gpu", "device_name", "sm"):
                if _worker(result)["hardware"][field] != _worker(source)["hardware"][field]:
                    raise ValueError("runtime hardware across phases differs from the accepted probe")
            expected, actual = _resource_values(source["resources"]), _resource_values(result["resources"])
            expected["runtime_memory"].pop("kv_cache_bytes")
            capacity = actual["runtime_memory"].pop("kv_cache_bytes")
            if actual != expected:
                raise ValueError("runtime cache layout or settings differ from the accepted probe")
            if capacity < bound:
                raise ValueError(f"runtime usable capacity {capacity} is smaller than the accepted bound {bound}")
            result["binding"] = deepcopy(binding)
            result["compatibility"] = {"status": "compatible", "runtime_settings_differences": settings_diff}
            phases[phase] = result
            inputs.extend([_file(plan_path), _file(checkpoint_path), *_evidence(index, result)])
            inputs.extend(_file(bundle.root / name) for name in bundle.files)
            inputs.extend([_file(bundle.manifest_path), _file(plan_path.parent / "generator-overrides.json")])
            config = plan.runtime_launch["model_config"]
            inputs.append(_file(config["path"], model_config=True))
            inputs.extend(
                _file(Path(config["path"]).parent / name, model_config=True) for name in config.get("source_files", {})
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, StopIteration) as error:
            raise ValueError(f"{phase}: {error}") from error
    resources = deepcopy(next(iter(phases.values()))["resources"])
    capacity = min(item["resources"]["runtime_memory"]["kv_cache_bytes"] for item in phases.values())
    provenance = {
        "source": "bound_collection_phases",
        "phases": {phase: item["provenance"] for phase, item in phases.items()},
    }
    resources["runtime_memory"].update(kv_cache_bytes=capacity, provenance=json.dumps(provenance, sort_keys=True))
    return {
        "status": "complete",
        "resources": resources,
        "phases": phases,
        "inputs": list({ref["path"]: ref for ref in inputs}.values()),
        "compatibility": {
            "status": "compatible",
            "scope": "memory compatibility only; collection-control review does not establish "
            "timing or measurement equivalence",
            "accepted_capacity_bytes": bound,
            "observed_capacity_bytes": capacity,
            "source_observations_index": manifest["observations_index"],
            "policy": "exact reviewed runtime changes; unchanged model, precision, effective settings and geometry; "
            "every phase/rank capacity at least the accepted bound",
        },
    }
