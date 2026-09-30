# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent native sweep repetitions with auditable measurement evidence.

AISimulate selects coordinates already measured by the source campaign. Dynamo
owns their admission and execution through its existing explicit-point API:
https://github.com/ai-dynamo/dynamo/blob/b83b1d9304ebfc624709ac46db32b1b6f1ff1615/components/src/dynamo/vllm/benchmark_points.py
No runtime scheduler implementation is copied or replaced here.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from aisimulate.fpm_contract import FPM_MANIFEST_FILENAME

from . import planner
from .capabilities import ModelCapabilityProfile, ResolvedDTypeProfile
from .config import FPMCollectionOptions, with_kv_warmup_defaults
from .execution_evidence import file_evidence, inspect_execution_evidence
from .measurement_evidence import compare_measurements, extract_measurement_evidence
from .memory_admission import DTypeMemoryEstimate, TopologyMemoryDecision
from .model_capability import ResolvedModelConfig
from .native_artifact import (
    NativeCollection,
    NativePointMeasurement,
    _rank_artifacts,
    select_native_measurements,
    validate_native_collection,
)
from .planner import BackendPolicy, FPMCollectionPlan, _canonical_hash, _hash_stable_admission
from .runner import (
    CHECKPOINT_SCHEMA,
    FPM_RECOVERABLE_STATUSES,
    _atomic_json,
    _cell_runner,
    _file_manifest,
    _recover_completed_attempt,
    _validate_points_receipts,
    run_collection,
)
from .runtime_memory import cell_from_dict, validate_saved_plan
from .types import ParallelTopology

PLAN_FILENAME = "repeatability-plan.json"
REPORT_FILENAME = "repeatability.json"
AGGREGATE_FILENAME = "repeatability-aggregate.json"
_LAUNCH_FILES = ("generator-request.json", "run.sh", "fpm_env.sh", "collector-runtime-env.sh")


class _IncompleteSample(ValueError):
    pass


def load_repeatability_deployment(source_campaign_dir: str | Path) -> dict[str, Any]:
    """Load the archived effective deployment inputs and verify their plan hash."""
    root = Path(source_campaign_dir).expanduser().resolve()
    path = root / "generator-overrides.json"
    if not path.is_file():
        raise ValueError(
            "source campaign lacks archived generator-overrides.json; supply the original explicit "
            "deployment inputs and verify their generator_config_sha256, or collect a new source campaign"
        )
    overrides = json.loads(path.read_text())
    saved = json.loads((root / "collection-plan.json").read_text())
    validate_saved_plan(saved)
    if not isinstance(overrides, dict):
        raise ValueError("archived deployment inputs must be an object")
    if not isinstance(overrides.get("K8sConfig", {}), dict):
        raise ValueError("archived K8sConfig must be an object")
    if _canonical_hash(with_kv_warmup_defaults(overrides)) != saved["generator_config_sha256"]:
        raise ValueError("archived deployment inputs differ from the source plan")
    return overrides


def load_repeatability_source(source_campaign_dir: str | Path) -> FPMCollectionPlan:
    """Load an exact archived runnable plan without resolving models or hardware.

    Deployment-only Generator inputs are supplied separately to run_repeatability
    and checked against the source's frozen generator_config_sha256. This loader
    never upgrades historical schemas or applies new planning defaults silently.
    """
    path = Path(source_campaign_dir).expanduser().resolve() / "collection-plan.json"
    saved = json.loads(path.read_text())
    validate_saved_plan(saved)
    if saved["schema_version"] != 11:
        raise ValueError(
            "repeatability execution requires a schema-v11 source campaign; historical data stays readable"
        )
    raw_options = saved["options"]
    option_names = {item.name for item in fields(FPMCollectionOptions)}
    options = {name: raw_options[name] for name in option_names if name in raw_options}
    for name, value in list(options.items()):
        if isinstance(value, list):
            options[name] = tuple(value)
    sampling = raw_options["prefill_sampling"]
    options.update(
        warmup_iterations=raw_options["global_warmup_iterations"],
        max_prefill_isl=sampling["max_isl"],
        max_prefill_batch_size=sampling["max_batch_size"],
        max_prefill_cudagraph_size=sampling["max_cudagraph_capture_size"],
        prefill_cudagraph_policy=sampling.get("cudagraph_policy", "explicit"),
    )
    if points := raw_options.get("benchmark_points"):
        options.update(
            benchmark_points_json=json.dumps(points["payload"], sort_keys=True, separators=(",", ":")),
            benchmark_points_sha256=points["sha256"],
        )
    capability = copy.deepcopy(saved["capability"])
    config = capability.pop("model_config")
    dtype = capability.pop("dtype")
    dtype["kv_cache_dtypes"] = tuple(dtype["kv_cache_dtypes"])
    resolved_dtype = ResolvedDTypeProfile(**dtype)
    resolved_capability = ModelCapabilityProfile(
        **capability,
        dtype=resolved_dtype,
        model_config=ResolvedModelConfig(
            config["payload"], source_kind=config["source_kind"], source_reference=str(path)
        ),
    )
    admissions = []
    for item in saved["topology_memory_admission"]:
        envelope = item["activation_envelope"]
        estimates = []
        for estimate in item["estimates"]:
            estimates.append(
                DTypeMemoryEstimate(**{key: value for key, value in estimate.items() if key != "headroom_bytes"})
            )
        admissions.append(
            TopologyMemoryDecision(
                topology=ParallelTopology(**item["topology"]),
                disposition=item["disposition"],
                source=item["source"],
                reason=item["reason"],
                estimates=tuple(estimates),
                max_new_tokens=envelope.get("requested_prefill_tokens", envelope.get("max_new_tokens")),
                max_batch_size=envelope["max_batch_size"],
                profile_max_num_tokens=envelope.get("max_num_tokens"),
            )
        )
    raw_dtype = dict(saved["dtype_profile"])
    raw_dtype["kv_cache_dtypes"] = tuple(raw_dtype["kv_cache_dtypes"])
    plan = FPMCollectionPlan(
        **{
            name: saved[name]
            for name in ("backend", "model_path", "system", "aic_revision", "generator_config_sha256", "sha256")
        },
        options=FPMCollectionOptions(**options),
        capability=resolved_capability,
        dtype_profile=ResolvedDTypeProfile(**raw_dtype),
        topologies=tuple(
            ParallelTopology(**{key: value for key, value in item.items() if key != "strategy"})
            for item in saved["topologies"]
        ),
        topology_memory_admission=tuple(admissions),
        backend_policies=tuple(BackendPolicy(**item) for item in saved["backend_policies"]),
        cells=tuple(cell_from_dict(item) for item in saved["cells"]),
        _fpm_profile_json=json.dumps(saved["fpm_profile"]) if "fpm_profile" in saved else None,
        _runtime_observation_json=json.dumps(saved["runtime_observation"]) if "runtime_observation" in saved else None,
    )
    if plan.to_dict() != saved:
        raise ValueError("saved repeatability source cannot be reconstructed exactly by this collector version")
    return plan


def _coordinates(point: dict[str, Any]) -> dict[str, Any]:
    coordinates = {key: point[key] for key in ("batch_size", "total_kv_read_tokens")}
    if point["point_type"] == "prefill":
        coordinates["total_prefill_tokens"] = point["total_prefill_tokens"]
        for key in ("partition", "rows"):
            if point.get(key) is not None:
                coordinates[key] = point[key]
    return coordinates


def _point_key(point: dict[str, Any]) -> str:
    return _canonical_hash(_coordinates(point))


def _regime(measurement: NativePointMeasurement) -> tuple[Any, ...]:
    point = measurement.point
    return (
        measurement.kv_seed_regime,
        point.get("expected_cudagraph_mode"),
        point.get("expected_capture_size"),
        point.get("padding_tokens"),
    )


def _select_points(collection: NativeCollection, limit: int | None, *, cell_id: str) -> list[dict[str, Any]]:
    """Cover extremes and regime/boundary representatives with a fixed budget."""
    points = sorted(
        select_native_measurements(collection, cell_id=cell_id),
        key=lambda item: (
            item.point["batch_size"],
            item.point["total_prefill_tokens"],
            item.point["total_kv_read_tokens"],
            _point_key(item.point),
        ),
    )
    if len({_point_key(item.point) for item in points}) != len(points):
        raise ValueError("repeatability requires unique native coordinates; duplicate regimes cannot be blended")
    roles: list[set[str]] = [set() for _ in points]
    axes = {
        "batch": [item.point["batch_size"] for item in points],
        "kv_per_request": [item.point["total_kv_read_tokens"] / item.point["batch_size"] for item in points],
        "new_tokens": [item.point["total_prefill_tokens"] for item in points],
    }
    for name, values in axes.items():
        for index, value in enumerate(values):
            for label, edge in (("minimum", min(values)), ("maximum", max(values))):
                if value == edge:
                    roles[index].add(f"{name}_{label}")
    for index, item in enumerate(points):
        roles[index].add(f"regime:{item.kv_seed_regime or 'unreported'}:{item.point.get('expected_cudagraph_mode')}")
    # Select the first and last observed capture boundary with neighbours that
    # actually exist in this grid. We do not invent new coordinates or promise
    # every capture size is represented by a bounded subset.
    token_axis = [item.point["total_prefill_tokens"] or item.point["batch_size"] for item in points]
    captures = sorted({value for item in points if (value := item.point.get("expected_capture_size")) is not None})
    for boundary in sorted(set(captures[:1] + captures[-1:])):
        for label, values in (
            ("below", [value for value in token_axis if value < boundary]),
            ("at", [value for value in token_axis if value == boundary]),
            ("above", [value for value in token_axis if value > boundary]),
        ):
            if not values:
                continue
            target = max(values) if label == "below" else min(values)
            for index, value in enumerate(token_axis):
                if value == target:
                    roles[index].add(f"capture_{boundary}_{label}")
    uncovered = set().union(*roles) if limit is not None else set()
    selected = set() if limit is not None else set(range(len(points)))
    while uncovered:
        index = max(range(len(points)), key=lambda index: (len(roles[index] & uncovered), -index))
        if len(selected) == limit:
            raise ValueError(
                f"repeatability max_points_per_cell={limit} cannot cover observed regimes/extremes; "
                f"increase the bound (uncovered: {sorted(uncovered)})"
            )
        selected.add(index)
        uncovered -= roles[index]
    # Fill remaining slots evenly across the sorted native grid. Every sample
    # remains a measured point, including when the source grid itself is small.
    target_count = min(limit, len(points)) if limit is not None else len(points)
    candidates = [round(index * (len(points) - 1) / max(1, target_count - 1)) for index in range(target_count)]
    for index in [*candidates, *range(len(points))]:
        if len(selected) >= target_count:
            break
        selected.add(index)
    return [
        {
            "key": _point_key(points[index].point),
            "coordinates": _coordinates(points[index].point),
            "source_point": points[index].point,
            "source_wall_time_seconds": max(value for _, value in points[index].rank_wall_times),
            "source_rank_wall_times": [list(value) for value in points[index].rank_wall_times],
            "kv_seed_regime": points[index].kv_seed_regime,
            "selection_reasons": sorted(roles[index]) if limit is not None else ["full_native_sweep"],
        }
        for index in sorted(selected)
    ]


def freeze_repeatability_plan(
    source_plan: FPMCollectionPlan,
    source_campaign_dir: str | Path,
    source_checkpoint_path: str | Path,
    *,
    samples: int = 5,
    max_points_per_cell: int = 12,
    cv_threshold: float = 0.05,
    comparison_mode: str = "full_grid",
    max_attempts_per_sample: int = 2,
    source_agreement_threshold: float = 0.05,
    observation_evidence_version: int = 2,
) -> dict[str, Any]:
    """Inspect validated source artifacts and return a deterministic frozen plan."""
    if type(observation_evidence_version) is not int or observation_evidence_version not in {1, 2}:
        raise ValueError("unsupported repeatability observation evidence version")
    if type(samples) is not int or samples < 2:
        raise ValueError("repeatability samples must be an integer >= 2")
    if type(max_points_per_cell) is not int or max_points_per_cell < 1:
        raise ValueError("repeatability max_points_per_cell must be a positive integer")
    if type(cv_threshold) not in (int, float) or not math.isfinite(cv_threshold) or cv_threshold < 0:
        raise ValueError("repeatability cv_threshold must be finite and nonnegative")
    if comparison_mode not in {"full_grid", "bounded"}:
        raise ValueError("repeatability comparison_mode must be full_grid or bounded")
    if type(max_attempts_per_sample) is not int or max_attempts_per_sample < 1:
        raise ValueError("repeatability max_attempts_per_sample must be a positive integer")
    if (
        type(source_agreement_threshold) not in (int, float)
        or not math.isfinite(source_agreement_threshold)
        or source_agreement_threshold < 0
    ):
        raise ValueError("repeatability source_agreement_threshold must be finite and nonnegative")
    root = Path(source_campaign_dir).expanduser().resolve()
    checkpoint_path = Path(source_checkpoint_path).expanduser().resolve()
    saved = json.loads((root / "collection-plan.json").read_text())
    validate_saved_plan(saved)
    if saved != source_plan.to_dict():
        raise ValueError("repeatability source plan differs from the saved campaign; reuse its exact frozen inputs")
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("plan_sha256") != source_plan.sha256:
        raise ValueError("repeatability source checkpoint does not match its campaign")
    cells = []
    for cell in source_plan.cells:
        entry = checkpoint.get("cells", {}).get(cell.cell_id, {})
        if entry.get("status") != "passed" or not entry.get("attempt_id"):
            raise ValueError(f"repeatability requires a passed source cell: {cell.cell_id}")
        cell_dir = root / "cells" / cell.cell_id
        collection = validate_native_collection(
            cell,
            cell_dir / "raw",
            expected_plan_sha256=source_plan.sha256,
            expected_attempt_id=entry["attempt_id"],
        )
        _validate_points_receipts(source_plan, cell, cell_dir / "raw", entry["attempt_id"])
        selected = _select_points(
            collection, max_points_per_cell if comparison_mode == "bounded" else None, cell_id=cell.cell_id
        )
        capture_sizes = sorted(
            {value for item in collection.points if (value := item.point.get("expected_capture_size")) is not None}
        )
        selected_boundaries = (
            sorted(set(capture_sizes[:1] + capture_sizes[-1:])) if comparison_mode == "bounded" else capture_sizes
        )
        manifest = {"schema_version": 3, "prefill": [], "decode": []}
        manifest[cell.workload_kind] = [item["coordinates"] for item in selected]
        cells.append(
            {
                "cell_id": cell.cell_id,
                "phase": cell.workload_kind,
                "source_attempt_id": collection.collector_attempt_id,
                "source_runtime_run_id": collection.runtime_run_id,
                "source_measurement": extract_measurement_evidence(
                    cell_dir / "raw", collection, evidence_version=observation_evidence_version
                ),
                "source_point_count": len(collection.points),
                "points": selected,
                "selection_coverage": {
                    "covered_strata": sorted({reason for point in selected for reason in point["selection_reasons"]}),
                    "uncovered_strata": [],
                    "observed_expected_capture_sizes": capture_sizes,
                    "selected_capture_boundaries": selected_boundaries,
                    "unselected_capture_boundaries": sorted(set(capture_sizes) - set(selected_boundaries)),
                },
                "benchmark_points": manifest,
                "source_files": _file_manifest(cell_dir / "raw"),
                "source_launch": {
                    name: file_evidence(cell_dir / name) if (cell_dir / name).is_file() else None
                    for name in _LAUNCH_FILES
                },
                "execution": inspect_execution_evidence(
                    cell, cell_dir / "raw", collection, plan=source_plan, evidence_version=observation_evidence_version
                ),
            }
        )
    payload = {
        **({"observation_evidence_version": 2} if observation_evidence_version == 2 else {}),
        "schema_name": "aisimulate_fpm_repeatability_plan",
        "schema_version": 2,
        "source_plan_sha256": source_plan.sha256,
        "source_campaign_dir": str(root),
        "source_checkpoint": file_evidence(checkpoint_path),
        "source_deployment": file_evidence(root / "generator-overrides.json")
        if (root / "generator-overrides.json").is_file()
        else None,
        "policy": {
            "samples": samples,
            "max_points_per_cell": max_points_per_cell,
            "cv_threshold": cv_threshold,
            "comparison_mode": comparison_mode,
            "max_attempts_per_sample": max_attempts_per_sample,
            "source_agreement_threshold": source_agreement_threshold,
        },
        "sampling": "independent_native_full_sweep_launches"
        if comparison_mode == "full_grid"
        else "independent_native_subset_launches",
        "cells": cells,
    }
    return {**payload, "sha256": _canonical_hash(payload)}


def _subset_plan(
    source: FPMCollectionPlan, selected: dict[str, Any], *, comparison_mode: str = "bounded"
) -> FPMCollectionPlan:
    canonical = json.dumps(selected["benchmark_points"], sort_keys=True, separators=(",", ":"))
    options = replace(
        source.options,
        benchmark_points_json=canonical,
        benchmark_points_sha256=_canonical_hash(selected["benchmark_points"]),
    )
    if comparison_mode == "full_grid":
        options = source.options
    plan = replace(
        source, options=options, cells=tuple(cell for cell in source.cells if cell.cell_id == selected["cell_id"])
    )
    # Use the ordinary collection-plan hash contract so existing native readers
    # can independently validate every sample campaign.
    payload = {
        "backend": plan.backend,
        "model_path": plan.model_path,
        "system": plan.system,
        "aic_revision": plan.aic_revision,
        "generator_config_sha256": plan.generator_config_sha256,
        "options": options.to_dict(),
        "capability": plan.capability.to_dict(),
        "dtype_profile": plan.dtype_profile.to_dict(),
        "point_generation": "dynamo_native_self_benchmark",
        "topology_memory_admission": [_hash_stable_admission(item) for item in plan.topology_memory_admission],
        "topologies": [item.to_dict() for item in plan.topologies],
        "policies": [item.to_dict() for item in plan.backend_policies],
        "cells": [cell.to_dict() for cell in plan.cells],
    }
    serialized = plan.to_dict()
    for key in ("fpm_profile", "runtime_memory_policy", "runtime_observation"):
        if key in serialized:
            payload[key] = serialized[key]
    return replace(plan, sha256=_canonical_hash(payload))


def _worker_identity(evidence):
    return sorted(
        [
            {
                key: worker[key]
                for key in (
                    "dp_rank",
                    "tp_rank",
                    "pp_rank",
                    "backend_version",
                    "attention_groups",
                    "graph_config",
                    "resolved_config",
                )
            }
            for worker in evidence["observed_workers"]
        ],
        key=lambda worker: (worker["dp_rank"], worker["tp_rank"], worker["pp_rank"]),
    )


def _sample_evidence(plan: FPMCollectionPlan, selected: dict[str, Any], directory: Path) -> dict[str, Any]:
    checkpoint = json.loads((directory / "checkpoint" / "fpm_forward.json").read_text())
    cell = plan.cells[0]
    entry = checkpoint.get("cells", {}).get(cell.cell_id, {})
    if entry.get("status") not in {"passed", "cleanup_failed", *FPM_RECOVERABLE_STATUSES} or not entry.get(
        "attempt_id"
    ):
        raise _IncompleteSample("repeatability sample collector checkpoint is not complete")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("plan_sha256") != plan.sha256:
        raise ValueError("repeatability sample collector checkpoint identity changed")
    root = directory / "artifacts" / plan.sha256[:16] / "cells" / cell.cell_id
    rank_payloads = _rank_artifacts(root / "raw")
    if not rank_payloads or len(rank_payloads) < cell.topology.dp:
        raise _IncompleteSample("repeatability sample is missing native rank artifacts")
    collection = validate_native_collection(
        cell,
        root / "raw",
        expected_plan_sha256=plan.sha256,
        expected_attempt_id=entry["attempt_id"],
    )
    _validate_points_receipts(plan, cell, root / "raw", entry["attempt_id"])
    canonical = select_native_measurements(collection, cell_id=cell.cell_id)
    actual = {_point_key(item.point): item for item in canonical}
    expected = {item["key"]: item for item in selected["points"]}
    if len(actual) != len(canonical) or set(actual) != set(expected):
        raise ValueError("repeatability runtime did not measure exactly the frozen coordinates")
    for key, item in actual.items():
        original = expected[key]
        original_measurement = NativePointMeasurement(
            point=original["source_point"],
            rank_wall_times=(),
            kv_seed_regime=original["kv_seed_regime"],
        )
        if _regime(item) != _regime(original_measurement):
            raise ValueError(f"repeatability execution/seed regime changed for point {key}")
    evidence_version = selected["source_measurement"]["schema_version"]
    execution = inspect_execution_evidence(cell, root / "raw", collection, plan=plan, evidence_version=evidence_version)
    if selected["execution"]["status"] == execution["status"] == "qualified" and _worker_identity(
        execution
    ) != _worker_identity(selected["execution"]):
        raise ValueError("repeatability observed attention backend, graph or runtime configuration changed")
    return {
        "collector_status": entry["status"],
        "collector_error": {key: entry[key] for key in ("error_type", "error") if key in entry},
        "cleanup_error": entry.get("cleanup_error"),
        "attempt_id": collection.collector_attempt_id,
        "runtime_run_id": collection.runtime_run_id,
        "runtime_grid_digest": collection.runtime_grid_digest,
        "measurement": extract_measurement_evidence(root / "raw", collection, evidence_version=evidence_version),
        "points": {
            key: {
                "wall_time_seconds": max(value for _, value in item.rank_wall_times),
                "rank_wall_times": [list(value) for value in item.rank_wall_times],
                "point": item.point,
                "kv_seed_regime": item.kv_seed_regime,
            }
            for key, item in actual.items()
        },
        "raw_files": _file_manifest(root / "raw"),
        "launch": {name: file_evidence(root / name) if (root / name).is_file() else None for name in _LAUNCH_FILES},
        "execution": execution,
    }


def _record_sample_evidence(plan, selected, directory, frozen, report, attempt):
    evidence = _sample_evidence(plan, selected, directory)
    used_attempts = {cell["source_attempt_id"] for cell in frozen["cells"]}
    used_runs = {cell["source_runtime_run_id"] for cell in frozen["cells"]}
    for entries in report["samples"].values():
        for entry in entries:
            for prior in entry["attempts"]:
                if prior.get("evidence"):
                    used_attempts.add(prior["evidence"]["attempt_id"])
                    used_runs.add(prior["evidence"]["runtime_run_id"])
    if evidence["runtime_run_id"] in used_runs:
        raise ValueError("repeatability reused a runtime run instead of an independent launch")
    if evidence["attempt_id"] in used_attempts:
        raise ValueError("repeatability reused a collector attempt instead of an independent launch")
    attempt["evidence"] = evidence
    if evidence["execution"]["status"] == "failed":
        attempt.update(
            status="failed",
            failure_kind="validation_failed",
            error="runtime execution evidence contradicts the frozen source",
        )
    elif evidence["collector_status"] == "cleanup_failed" or evidence["cleanup_error"] is not None:
        attempt.update(status="failed", failure_kind="cleanup_failed", error=evidence["cleanup_error"])
    elif evidence["collector_status"] in FPM_RECOVERABLE_STATUSES:
        attempt.update(
            status="failed",
            failure_kind="postprocessing_failed",
            error="complete native measurement retained; CPU-only post-processing recovery is required before "
            "qualification or further launches; GPU replacement is disabled",
        )
    elif not attempt.get("errors"):
        attempt["status"] = "passed"


def _retry_sample_cleanup(plan, selected, directory, attempt):
    """Retry teardown alone, preserving the original measurement and checkpoint."""
    cell_root = directory / "artifacts" / plan.sha256[:16] / "cells" / selected["cell_id"]
    cleanup = {"status": "running"}
    attempt.setdefault("cleanup_attempts", []).append(cleanup)
    try:
        manifest = cell_root / FPM_MANIFEST_FILENAME
        if not manifest.is_file():
            raise ValueError("repeatability sample has no manifest to verify cleanup")
        _cell_runner(plan, plan.cells[0], manifest, cell_root).cleanup()
        if _sample_evidence(plan, selected, directory) != attempt["evidence"]:
            raise ValueError("repeatability sample evidence changed during cleanup")
        cleanup["status"] = "passed"
        if attempt["evidence"]["collector_status"] in FPM_RECOVERABLE_STATUSES:
            attempt.update(status="failed", failure_kind="postprocessing_failed")
        else:
            attempt["status"] = "passed"
    except (KeyboardInterrupt, SystemExit):
        cleanup["status"] = "interrupted"
        raise
    except Exception as error:
        cleanup.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        attempt.setdefault("cleanup_file_history", []).append(attempt.get("files", {}))
        attempt["files"] = _file_manifest(directory)


def _retry_sample_postprocessing(plan, selected, directory, attempt):
    """Validate salvaged artifacts without relaunching or rewriting their checkpoint."""
    recovery = {"status": "running"}
    attempt.setdefault("postprocessing_attempts", []).append(recovery)
    try:
        checkpoint = json.loads((directory / "checkpoint" / "fpm_forward.json").read_text())
        entry = dict(checkpoint["cells"][selected["cell_id"]])
        if attempt.get("cleanup_attempts") and attempt["cleanup_attempts"][-1]["status"] == "passed":
            entry.pop("cleanup_error", None)
        recovered = _recover_completed_attempt(plan, plan.cells[0], directory / "artifacts" / plan.sha256[:16], entry)
        if recovered is None:
            raise ValueError(
                "CPU-only recovery could not validate the retained observation; inspect collector artifacts "
                "and teardown before retrying recovery; the original GPU measurement cannot be replaced"
            )
        if _sample_evidence(plan, selected, directory) != attempt["evidence"]:
            raise ValueError("repeatability sample evidence changed during post-processing recovery")
        recovery.update(status="passed", artifact_recovery=recovered["artifact_recovery"])
        attempt["status"] = "passed"
    except (KeyboardInterrupt, SystemExit):
        recovery["status"] = "interrupted"
        raise
    except Exception as error:
        recovery.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        attempt.setdefault("postprocessing_file_history", []).append(attempt.get("files", {}))
        attempt["files"] = _file_manifest(directory)


def _comparison_status(comparisons: list[dict[str, Any]]) -> str:
    if any(item["status"] == "mismatch" for item in comparisons):
        return "mismatch"
    if not comparisons or any(item["status"] != "comparable" for item in comparisons):
        return "unestablished"
    return "comparable"


def _summarize(frozen: dict[str, Any], report: dict[str, Any]) -> None:
    """Keep independent observations, identity eligibility and stability separate."""
    points = []
    required = frozen["policy"]["samples"]
    full_grid = frozen["policy"]["comparison_mode"] == "full_grid"
    execution_failed = any(cell["execution"]["status"] == "failed" for cell in frozen["cells"])
    execution_complete = True
    validation_failed = False
    for cell in frozen["cells"]:
        slots = report["samples"][cell["cell_id"]]
        latest = [sample["attempts"][-1] for sample in slots if sample["attempts"]]
        validation_failed |= any(item.get("failure_kind") == "validation_failed" for item in latest)
        # A valid observation is never superseded. Preserve even a completed
        # measurement whose collection subsequently failed during teardown.
        measured = [attempt for slot in slots for attempt in slot["attempts"] if attempt.get("evidence")]
        execution_failed |= any(item["evidence"]["execution"]["status"] == "failed" for item in measured)
        execution_complete &= len(measured) == required and all(
            item["status"] == "passed" and item["evidence"]["execution"]["status"] == "qualified" for item in measured
        )
        reference = measured[0]["evidence"].get("measurement", {}) if measured else {}
        for point in cell["points"]:
            key = point["key"]
            comparisons = []
            if measured:
                first_point = reference.get("points", {}).get(key, {})
                first_recorded = (
                    not reference.get("reasons")
                    and first_point.get("status") == "recorded"
                    and measured[0]["evidence"]["execution"]["status"] == "qualified"
                )
                comparisons.append(
                    {
                        "status": "mismatch"
                        if measured[0]["evidence"]["execution"]["status"] == "failed"
                        else "comparable"
                        if first_recorded
                        else "unestablished",
                        "reasons": [*reference.get("reasons", []), *first_point.get("reasons", [])],
                        "scope": "reference observation; independence checked by launch identities",
                    }
                )
                for item in measured[1:]:
                    comparison = compare_measurements(reference, item["evidence"].get("measurement", {}), point_key=key)
                    if item["evidence"]["execution"]["status"] != "qualified":
                        comparison = {
                            **comparison,
                            "status": "mismatch"
                            if item["evidence"]["execution"]["status"] == "failed"
                            else "unestablished",
                            "reasons": [*comparison["reasons"], "worker execution is not qualified"],
                        }
                    if (
                        measured[0]["evidence"]["execution"]["status"] == "qualified"
                        and item["evidence"]["execution"]["status"] == "qualified"
                        and _worker_identity(measured[0]["evidence"]["execution"])
                        != _worker_identity(item["evidence"]["execution"])
                    ):
                        comparison = {
                            **comparison,
                            "status": "mismatch",
                            "reasons": [*comparison["reasons"], "observed worker runtime configuration differs"],
                        }
                    comparisons.append(comparison)
            comparable = _comparison_status(comparisons)
            source_comparison = compare_measurements(
                cell["source_measurement"], reference, point_key=key, same_context=full_grid
            )
            if cell["execution"]["status"] != "qualified":
                source_comparison = {
                    **source_comparison,
                    "status": "mismatch" if cell["execution"]["status"] == "failed" else "unestablished",
                    "reasons": [*source_comparison["reasons"], "source worker execution is not qualified"],
                }
            raw_values = [item["evidence"]["points"][key]["wall_time_seconds"] for item in measured]
            # Raw timings remain visible even when identity cannot be established;
            # they must not be pooled into a qualified numerical population.
            values = raw_values if comparable == "comparable" else []
            mean = statistics.mean(values) if values else None
            median = statistics.median(values) if values else None
            stddev = statistics.stdev(values) if len(values) >= 2 else None
            cv = stddev / mean if stddev is not None else None
            inclusive = [point["source_wall_time_seconds"], *values]
            inclusive_stddev = statistics.stdev(inclusive) if values else None
            inclusive_cv = inclusive_stddev / statistics.mean(inclusive) if inclusive_stddev is not None else None
            difference = (
                abs(median / point["source_wall_time_seconds"] - 1)
                if median is not None and source_comparison["status"] == "comparable"
                else None
            )
            status = (
                "mismatch"
                if comparable == "mismatch"
                else "unestablished"
                if comparable != "comparable"
                else "incomplete"
                if len(values) != required
                else "passed"
                if cv is not None and cv <= frozen["policy"]["cv_threshold"]
                else "unstable"
            )
            points.append(
                {
                    "cell_id": cell["cell_id"],
                    "phase": cell["phase"],
                    "key": key,
                    "coordinates": point["coordinates"],
                    "source_wall_time_seconds": point["source_wall_time_seconds"],
                    "raw_samples_seconds": raw_values,
                    "samples_seconds": values,
                    "sample_count": len(values),
                    "raw_sample_count": len(raw_values),
                    "observations": [
                        {
                            "attempt_id": item["evidence"]["attempt_id"],
                            "runtime_run_id": item["evidence"]["runtime_run_id"],
                            "directory": item["directory"],
                            "status": item["status"],
                            **item["evidence"]["points"][key],
                        }
                        for item in measured
                    ],
                    "comparability": {"status": comparable, "comparisons": comparisons},
                    "source_comparison": source_comparison,
                    "mean_seconds": mean,
                    "median_seconds": median,
                    "minimum_seconds": min(values) if values else None,
                    "maximum_seconds": max(values) if values else None,
                    "sample_cv": cv,
                    "sample_stddev_seconds": stddev,
                    "source_inclusive_cv": inclusive_cv,
                    "source_inclusive_stddev_seconds": inclusive_stddev,
                    "source_inclusive_sample_count": len(inclusive),
                    "source_inclusive_usage": "diagnostic_only; source is not an independent fresh repetition",
                    "source_relative_difference": difference,
                    "source_agreement": (
                        "unestablished"
                        if difference is None or len(values) != required
                        else "passed"
                        if difference <= frozen["policy"]["source_agreement_threshold"]
                        else "failed"
                    ),
                    "status": status,
                }
            )
    counts = Counter(item["status"] for item in points)
    repeatability = (
        "unstable"
        if counts["unstable"]
        else "mismatch"
        if counts["mismatch"]
        else "unestablished"
        if counts["unestablished"]
        else "incomplete"
        if counts["incomplete"]
        else "passed"
    )
    new_status = (
        "unstable"
        if repeatability == "unstable"
        else "failed"
        if execution_failed or validation_failed or counts["mismatch"]
        else "qualified"
        if repeatability == "passed" and execution_complete
        else "unestablished"
        if repeatability == "unestablished"
        else "incomplete"
    )
    source_status = (
        "failed"
        if any(
            point["source_comparison"]["status"] == "mismatch" or point["source_agreement"] == "failed"
            for point in points
        )
        else "qualified"
        if full_grid
        and new_status == "qualified"
        and all(
            point["source_comparison"]["status"] == "comparable" and point["source_agreement"] == "passed"
            for point in points
        )
        else "unestablished"
    )
    report.update(
        points=points,
        repeatability={
            "status": repeatability,
            "cv_threshold": frozen["policy"]["cv_threshold"],
            "point_counts": dict(counts),
            "measurement_count": "independent fresh launches; adjacent internal steps are not repetitions",
            "time_unit": "seconds",
            "standard_deviation_denominator": "n - 1",
        },
        new_population={"status": new_status, "scope": "full_native_sweep" if full_grid else "bounded_diagnostic"},
        source_qualification={
            "status": source_status,
            "source_agreement_threshold": frozen["policy"]["source_agreement_threshold"],
            "agreement_metric": "abs(fresh_median / source_wall_time - 1)",
            "scope": "original_source" if full_grid else "bounded_subset_cannot_qualify_full_source",
        },
        execution={
            "status": "failed"
            if execution_failed or validation_failed
            else "qualified"
            if execution_complete
            else "incomplete"
        },
        status=(
            "failed"
            if new_status in {"failed", "unstable"} or source_status == "failed"
            else "passed"
            if new_status == source_status == "qualified"
            else "incomplete"
        ),
    )


def _save_report(root: Path, report: dict[str, Any]) -> None:
    """Publish an auditable derived estimate without replacing formal source data."""
    aggregate = {
        "schema_name": "aisimulate_fpm_repeatability_aggregate",
        "schema_version": 1,
        "plan_sha256": report["plan_sha256"],
        "status": report["new_population"]["status"],
        "scope": report["new_population"]["scope"],
        "method": "median_of_independent_launch_max_rank_estimates",
        "formal_source_replaced": False,
        "points": [
            {
                key: point[key]
                for key in (
                    "cell_id",
                    "phase",
                    "key",
                    "coordinates",
                    "median_seconds",
                    "samples_seconds",
                    "raw_samples_seconds",
                    "sample_cv",
                    "status",
                    "comparability",
                    "observations",
                )
            }
            for point in report["points"]
        ],
    }
    # The durable report must keep referencing its previous immutable aggregate
    # until the new report is committed. An interruption cannot orphan its hash.
    aggregate_path = root / f"{Path(AGGREGATE_FILENAME).stem}-{_canonical_hash(aggregate)}.json"
    if aggregate_path.exists():
        if json.loads(aggregate_path.read_text()) != aggregate:
            raise ValueError("repeatability aggregate generation changed")
    else:
        _atomic_json(aggregate_path, aggregate)
    report["aggregate"] = file_evidence(aggregate_path)
    _atomic_json(root / REPORT_FILENAME, report)


def assess_repeatability(
    frozen_plan: dict[str, Any],
    report: dict[str, Any],
    *,
    cv_threshold: float,
    source_agreement_threshold: float | None = None,
) -> dict[str, Any]:
    """Reassess preserved samples without scheduling any additional collection.

    The returned assessment records the original frozen plan and the changed
    criterion. A caller must retain it at a new report path, preserving history.
    Source and raw sample evidence are revalidated before any reassessment.
    """
    if type(cv_threshold) not in (int, float) or not math.isfinite(cv_threshold) or cv_threshold < 0:
        raise ValueError("repeatability cv_threshold must be finite and nonnegative")
    if report.get("plan_sha256") != frozen_plan.get("sha256"):
        raise ValueError("repeatability report does not match its frozen plan")
    payload = {key: value for key, value in frozen_plan.items() if key != "sha256"}
    if _canonical_hash(payload) != frozen_plan.get("sha256"):
        raise ValueError("repeatability frozen plan hash changed")
    if frozen_plan.get("schema_version") == 1:
        return _assess_legacy(frozen_plan, report, cv_threshold=cv_threshold)
    if frozen_plan.get("schema_version") != 2 or report.get("schema_version") != 2:
        raise ValueError("unsupported repeatability evidence schema")
    source = load_repeatability_source(frozen_plan["source_campaign_dir"])
    fresh = freeze_repeatability_plan(
        source,
        frozen_plan["source_campaign_dir"],
        frozen_plan["source_checkpoint"]["path"],
        observation_evidence_version=frozen_plan.get("observation_evidence_version", 1),
        **frozen_plan["policy"],
    )
    if fresh != frozen_plan:
        raise ValueError("repeatability source artifacts changed before reassessment")
    _validate_saved_samples(source, frozen_plan, report, Path(report["output_dir"]).resolve())
    result = copy.deepcopy(report)
    assessment_plan = copy.deepcopy(frozen_plan)
    assessment_plan["policy"]["cv_threshold"] = cv_threshold
    if source_agreement_threshold is not None:
        if (
            type(source_agreement_threshold) not in (int, float)
            or not math.isfinite(source_agreement_threshold)
            or source_agreement_threshold < 0
        ):
            raise ValueError("repeatability source_agreement_threshold must be finite and nonnegative")
        assessment_plan["policy"]["source_agreement_threshold"] = source_agreement_threshold
    result["assessment"] = {
        "source_report_sha256": _canonical_hash(report),
        "cv_threshold": cv_threshold,
        "source_agreement_threshold": assessment_plan["policy"]["source_agreement_threshold"],
    }
    _summarize(assessment_plan, result)
    return result


def _assess_legacy(frozen, report, *, cv_threshold):
    """Keep historical bounded reports inspectable without a protocol upgrade."""
    root = Path(frozen["source_campaign_dir"]).resolve()
    if file_evidence(Path(frozen["source_checkpoint"]["path"])) != frozen["source_checkpoint"]:
        raise ValueError("repeatability historical source checkpoint changed")
    for cell in frozen["cells"]:
        if _file_manifest(root / "cells" / cell["cell_id"] / "raw") != cell["source_files"]:
            raise ValueError("repeatability historical source artifacts changed")
    output = Path(report["output_dir"]).resolve()
    for samples in report["samples"].values():
        for sample in samples:
            for attempt in sample["attempts"]:
                path = (output / attempt["directory"]).resolve()
                if not path.is_relative_to(output) or _file_manifest(path) != attempt.get("files"):
                    raise ValueError("repeatability historical sample artifacts changed")
    result = copy.deepcopy(report)
    result["historical_assessment"] = {
        key: report[key] for key in ("status", "repeatability", "execution") if key in report
    }
    result["assessment"] = {
        "source_report_sha256": _canonical_hash(report),
        "cv_threshold": cv_threshold,
        "reason": "legacy bounded measurements lack a frozen comparable protocol",
    }
    result["status"] = "incomplete"
    result["source_qualification"] = {"status": "unestablished"}
    result["new_population"] = {"status": "unestablished", "scope": "legacy_diagnostic"}
    return result


def _validate_saved_samples(source, frozen, report, root):
    if report.get("schema_version") != 2:
        raise ValueError("legacy repeatability evidence cannot resume execution; preserve it for offline inspection")
    if report.get("aggregate") and file_evidence(Path(report["aggregate"]["path"])) != report["aggregate"]:
        raise ValueError("repeatability aggregate artifact changed")
    if report.get("output_dir") != str(root):
        raise ValueError("repeatability report output directory changed")
    if report.get("policy") != frozen["policy"]:
        raise ValueError("repeatability saved measurement policy changed")
    expected_cells = {cell["cell_id"] for cell in frozen["cells"]}
    if set(report.get("samples", {})) != expected_cells:
        raise ValueError("repeatability saved sample cells do not match the plan")
    seen_attempts = {cell["source_attempt_id"] for cell in frozen["cells"]}
    seen_runs = {cell["source_runtime_run_id"] for cell in frozen["cells"]}
    for cell in frozen["cells"]:
        samples = report["samples"][cell["cell_id"]]
        if len(samples) != frozen["policy"]["samples"]:
            raise ValueError("repeatability saved sample count changed")
        plan = _subset_plan(source, cell, comparison_mode=frozen["policy"]["comparison_mode"])
        for index, sample in enumerate(samples, start=1):
            if sample.get("sample_index") != index or not isinstance(sample.get("attempts"), list):
                raise ValueError("repeatability saved sample indices are invalid")
            if len(sample["attempts"]) > frozen["policy"]["max_attempts_per_sample"]:
                raise ValueError("repeatability saved attempts exceed the frozen retry budget")
            for attempt_index, attempt in enumerate(sample["attempts"], start=1):
                relative = Path("samples") / cell["cell_id"] / f"sample-{index:02d}" / f"attempt-{attempt_index:02d}"
                directory = root / relative
                if attempt.get("directory") != str(relative) or not directory.resolve().is_relative_to(root):
                    raise ValueError("repeatability saved attempt directory is invalid")
                if attempt.get("status") not in {"running", "interrupted", "failed", "passed"}:
                    raise ValueError("repeatability saved attempt status is invalid")
                if attempt.get("files") is not None and _file_manifest(directory) != attempt["files"]:
                    raise ValueError("repeatability attempt files changed")
                if attempt["status"] != "passed" and not attempt.get("evidence"):
                    continue
                if attempt_index != len(sample["attempts"]):
                    raise ValueError("repeatability valid measurement was superseded")
                evidence = _sample_evidence(plan, cell, directory)
                if evidence != attempt.get("evidence"):
                    raise ValueError("saved repeatability sample evidence changed")
                if (
                    attempt["status"] == "passed"
                    and (evidence["collector_status"] == "cleanup_failed" or evidence["cleanup_error"] is not None)
                    and not (attempt.get("cleanup_attempts") and attempt["cleanup_attempts"][-1]["status"] == "passed")
                ):
                    raise ValueError("repeatability sample cleanup is not resolved")
                if (
                    attempt["status"] == "passed"
                    and evidence["collector_status"] in FPM_RECOVERABLE_STATUSES
                    and not (
                        attempt.get("postprocessing_attempts")
                        and attempt["postprocessing_attempts"][-1]["status"] == "passed"
                    )
                ):
                    raise ValueError("repeatability sample post-processing is not resolved")
                if evidence["attempt_id"] in seen_attempts:
                    raise ValueError("repeatability reused an attempt instead of an independent launch")
                if evidence["runtime_run_id"] in seen_runs:
                    raise ValueError("repeatability reused a runtime run instead of an independent launch")
                seen_attempts.add(evidence["attempt_id"])
                seen_runs.add(evidence["runtime_run_id"])


def run_repeatability(
    source_plan: FPMCollectionPlan,
    *,
    generator_overrides: dict[str, Any],
    source_campaign_dir: str | Path,
    source_checkpoint_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    retry_failed: bool = False,
    samples: int = 5,
    max_points_per_cell: int = 12,
    cv_threshold: float = 0.05,
    comparison_mode: str = "full_grid",
    max_attempts_per_sample: int = 2,
    source_agreement_threshold: float = 0.05,
    observation_evidence_version: int | None = None,
) -> dict[str, Any]:
    """Repeat the original sweep, or collect an explicitly diagnostic bounded subset.

    The caller owns execution authorization and an allocation for the selected
    Kubernetes or Slurm transport. No serving data or formal table is replaced.
    """
    if retry_failed and not resume:
        raise ValueError("repeatability retry_failed requires resume")
    if _canonical_hash(with_kv_warmup_defaults(generator_overrides)) != source_plan.generator_config_sha256:
        raise ValueError("repeatability deployment inputs differ from the original launch")
    root = Path(output_dir).expanduser().resolve()
    previous = json.loads((root / PLAN_FILENAME).read_text()) if resume and (root / PLAN_FILENAME).exists() else None
    if observation_evidence_version is None:
        observation_evidence_version = previous.get("observation_evidence_version", 1) if previous is not None else 2
    frozen = freeze_repeatability_plan(
        source_plan,
        source_campaign_dir,
        source_checkpoint_path,
        samples=samples,
        max_points_per_cell=max_points_per_cell,
        cv_threshold=cv_threshold,
        comparison_mode=comparison_mode,
        max_attempts_per_sample=max_attempts_per_sample,
        source_agreement_threshold=source_agreement_threshold,
        observation_evidence_version=observation_evidence_version,
    )
    source = Path(source_campaign_dir).expanduser().resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("repeatability output must be separate from the source campaign")
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise ValueError("repeatability output exists; use resume or a fresh directory")
        if json.loads((root / PLAN_FILENAME).read_text()) != frozen:
            raise ValueError("repeatability frozen plan or source artifacts changed")
        report = json.loads((root / REPORT_FILENAME).read_text())
        if report.get("plan_sha256") != frozen["sha256"]:
            raise ValueError("repeatability report does not match its frozen plan")
        _validate_saved_samples(source_plan, frozen, report, root)
        _summarize(frozen, report)
    else:
        if resume:
            raise ValueError("repeatability resume requires existing validation evidence")
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / PLAN_FILENAME, frozen)
        report = {
            "schema_name": "aisimulate_fpm_repeatability",
            "schema_version": 2,
            "plan_sha256": frozen["sha256"],
            "source_plan_sha256": source_plan.sha256,
            "output_dir": str(root),
            "policy": frozen["policy"],
            "samples": {
                cell["cell_id"]: [{"sample_index": index + 1, "attempts": []} for index in range(samples)]
                for cell in frozen["cells"]
            },
        }
        _summarize(frozen, report)
        _save_report(root, report)
    if any(cell["execution"]["status"] == "failed" for cell in frozen["cells"]):
        # A stable repeat set cannot qualify a measured wrong-source campaign.
        # Retain its assessment without spending another benchmark launch.
        _atomic_json(root / REPORT_FILENAME, report)
        return report
    for selected in frozen["cells"]:
        plan = _subset_plan(source_plan, selected, comparison_mode=comparison_mode)
        for sample in report["samples"][selected["cell_id"]]:
            previous = sample["attempts"][-1] if sample["attempts"] else None
            if previous is not None:
                previous_dir = root / previous["directory"]
                if previous["status"] != "passed" and not previous.get("evidence"):
                    # Collection can finish before the final report is committed.
                    # Recover its verified observation without launching a replacement.
                    try:
                        _record_sample_evidence(plan, selected, previous_dir, frozen, report, previous)
                    except (FileNotFoundError, _IncompleteSample):
                        pass
                    except ValueError as error:
                        previous.update(status="failed", failure_kind="validation_failed", error=str(error))
                    previous["files"] = _file_manifest(previous_dir)
                    _summarize(frozen, report)
                    _save_report(root, report)
                if previous["status"] == "passed":
                    continue
                if previous.get("evidence"):
                    if retry_failed and previous.get("failure_kind") == "cleanup_failed":
                        try:
                            _retry_sample_cleanup(plan, selected, previous_dir, previous)
                        finally:
                            _summarize(frozen, report)
                            _save_report(root, report)
                        if previous["status"] == "passed":
                            continue
                    if retry_failed and previous.get("failure_kind") == "postprocessing_failed":
                        try:
                            _retry_sample_postprocessing(plan, selected, previous_dir, previous)
                        finally:
                            _summarize(frozen, report)
                            _save_report(root, report)
                        if previous["status"] == "passed":
                            continue
                    return report
                if not retry_failed:
                    return report
                if len(sample["attempts"]) >= max_attempts_per_sample:
                    raise ValueError("repeatability frozen retry budget exhausted; retain attempts and investigate")
                # A failed teardown must be resolved before any next launch.
                manifest = (
                    previous_dir
                    / "artifacts"
                    / plan.sha256[:16]
                    / "cells"
                    / selected["cell_id"]
                    / FPM_MANIFEST_FILENAME
                )
                if manifest.exists():
                    try:
                        _cell_runner(plan, plan.cells[0], manifest, manifest.parent).cleanup()
                    finally:
                        # Teardown failures may append transport diagnostics;
                        # retain the old snapshot and record the appended files.
                        previous.setdefault("cleanup_file_history", []).append(previous.get("files", {}))
                        previous["files"] = _file_manifest(previous_dir)
                        _atomic_json(root / REPORT_FILENAME, report)
            relative = (
                Path("samples")
                / selected["cell_id"]
                / f"sample-{sample['sample_index']:02d}"
                / f"attempt-{len(sample['attempts']) + 1:02d}"
            )
            if planner._git_revision() != source_plan.aic_revision:
                raise ValueError(
                    "repeatability launch requires the source collector revision; the current checkout/code "
                    "is unqualified for this campaign. Use the matching revision or collect a new source campaign. "
                    "Existing samples remain available for offline assessment."
                )
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=False)
            attempt = {"directory": str(relative), "status": "running"}
            sample["attempts"].append(attempt)
            _atomic_json(root / REPORT_FILENAME, report)
            try:
                validating = False
                errors = run_collection(
                    plan,
                    generator_overrides=generator_overrides,
                    checkpoint_dir=str(directory / "checkpoint"),
                    artifact_root=str(directory / "artifacts"),
                    resume=False,
                    retry_failed=False,
                    publish_database=False,
                )
                if errors:
                    attempt.update(status="failed", errors=errors, failure_kind="collection_error")
                validating = True
                _record_sample_evidence(plan, selected, directory, frozen, report, attempt)
            except (KeyboardInterrupt, SystemExit):
                attempt.update(status="interrupted", failure_kind="interrupted")
                raise
            except Exception as error:
                kind = (
                    "missing_evidence"
                    if isinstance(error, (FileNotFoundError, _IncompleteSample))
                    else "validation_failed"
                    if validating and isinstance(error, ValueError)
                    else "collection_error"
                )
                attempt.update(status="failed", failure_kind=kind, error=f"{type(error).__name__}: {error}")
            finally:
                attempt["files"] = _file_manifest(directory)
                _summarize(frozen, report)
                _save_report(root, report)
            if (
                attempt["status"] != "passed"
                or report["repeatability"]["status"] == "mismatch"
                or report["source_qualification"]["status"] == "failed"
            ):
                # Keep failures visible. Do not launch the remaining subset
                # repetitions into an uninvestigated runtime/cleanup failure.
                return report
    _summarize(frozen, report)
    _save_report(root, report)
    return report
