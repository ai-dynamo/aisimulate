# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, independent native remeasurement of a validated collection.

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
    _atomic_json,
    _cell_runner,
    _file_manifest,
    _validate_points_receipts,
    run_collection,
)
from .runtime_memory import cell_from_dict, validate_saved_plan
from .types import ParallelTopology

PLAN_FILENAME = "repeatability-plan.json"
REPORT_FILENAME = "repeatability.json"
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


def _select_points(collection: NativeCollection, limit: int, *, cell_id: str) -> list[dict[str, Any]]:
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
    uncovered = set().union(*roles)
    selected = set()
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
    target_count = min(limit, len(points))
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
            "selection_reasons": sorted(roles[index]),
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
) -> dict[str, Any]:
    """Inspect validated source artifacts and return a deterministic frozen plan."""
    if type(samples) is not int or samples < 2:
        raise ValueError("repeatability samples must be an integer >= 2")
    if type(max_points_per_cell) is not int or max_points_per_cell < 1:
        raise ValueError("repeatability max_points_per_cell must be a positive integer")
    if type(cv_threshold) not in (int, float) or not math.isfinite(cv_threshold) or cv_threshold < 0:
        raise ValueError("repeatability cv_threshold must be finite and nonnegative")
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
        selected = _select_points(collection, max_points_per_cell, cell_id=cell.cell_id)
        capture_sizes = sorted(
            {value for item in collection.points if (value := item.point.get("expected_capture_size")) is not None}
        )
        selected_boundaries = sorted(set(capture_sizes[:1] + capture_sizes[-1:]))
        manifest = {"schema_version": 3, "prefill": [], "decode": []}
        manifest[cell.workload_kind] = [item["coordinates"] for item in selected]
        cells.append(
            {
                "cell_id": cell.cell_id,
                "phase": cell.workload_kind,
                "source_attempt_id": collection.collector_attempt_id,
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
                "execution": inspect_execution_evidence(cell, cell_dir / "raw", collection, plan=source_plan),
            }
        )
    payload = {
        "schema_name": "aisimulate_fpm_repeatability_plan",
        "schema_version": 1,
        "source_plan_sha256": source_plan.sha256,
        "source_campaign_dir": str(root),
        "source_checkpoint": file_evidence(checkpoint_path),
        "source_deployment": file_evidence(root / "generator-overrides.json")
        if (root / "generator-overrides.json").is_file()
        else None,
        "policy": {"samples": samples, "max_points_per_cell": max_points_per_cell, "cv_threshold": cv_threshold},
        "sampling": "independent_native_subset_launches",
        "cells": cells,
    }
    return {**payload, "sha256": _canonical_hash(payload)}


def _subset_plan(source: FPMCollectionPlan, selected: dict[str, Any]) -> FPMCollectionPlan:
    canonical = json.dumps(selected["benchmark_points"], sort_keys=True, separators=(",", ":"))
    options = replace(
        source.options,
        benchmark_points_json=canonical,
        benchmark_points_sha256=_canonical_hash(selected["benchmark_points"]),
    )
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


def _sample_evidence(plan: FPMCollectionPlan, selected: dict[str, Any], directory: Path) -> dict[str, Any]:
    checkpoint = json.loads((directory / "checkpoint" / "fpm_forward.json").read_text())
    cell = plan.cells[0]
    entry = checkpoint.get("cells", {}).get(cell.cell_id, {})
    if entry.get("status") != "passed" or not entry.get("attempt_id"):
        raise _IncompleteSample("repeatability sample collector checkpoint is not complete")
    if checkpoint.get("plan_sha256") != plan.sha256:
        raise ValueError("repeatability sample collector checkpoint is not passed")
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
    actual = {_point_key(item.point): item for item in collection.points}
    expected = {item["key"]: item for item in selected["points"]}
    if len(actual) != len(collection.points) or set(actual) != set(expected):
        raise ValueError("repeatability runtime did not measure exactly the frozen subset")
    for key, item in actual.items():
        original = expected[key]
        original_measurement = NativePointMeasurement(
            point=original["source_point"],
            rank_wall_times=(),
            kv_seed_regime=original["kv_seed_regime"],
        )
        if _regime(item) != _regime(original_measurement):
            raise ValueError(f"repeatability execution/seed regime changed for point {key}")
    execution = inspect_execution_evidence(cell, root / "raw", collection, plan=plan)
    if selected["execution"]["status"] == "qualified" and execution["status"] == "qualified":

        def observed_values(evidence):
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

        if observed_values(execution) != observed_values(selected["execution"]):
            raise ValueError("repeatability observed attention backend, graph or runtime configuration changed")
    return {
        "attempt_id": collection.collector_attempt_id,
        "runtime_run_id": collection.runtime_run_id,
        "runtime_grid_digest": collection.runtime_grid_digest,
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


def _summarize(frozen: dict[str, Any], report: dict[str, Any]) -> None:
    points = []
    execution_complete = all(cell["execution"]["status"] == "qualified" for cell in frozen["cells"])
    execution_failed = any(cell["execution"]["status"] == "failed" for cell in frozen["cells"])
    validation_failed = False
    required = frozen["policy"]["samples"]
    for cell in frozen["cells"]:
        latest_attempts = [
            sample["attempts"][-1] for sample in report["samples"][cell["cell_id"]] if sample["attempts"]
        ]
        validation_failed |= any(attempt.get("failure_kind") == "validation_failed" for attempt in latest_attempts)
        execution_failed |= any(
            attempt.get("evidence", {}).get("execution", {}).get("status") == "failed" for attempt in latest_attempts
        )
        successful = [
            sample["attempts"][-1]
            for sample in report["samples"][cell["cell_id"]]
            if sample["attempts"] and sample["attempts"][-1]["status"] == "passed"
        ]
        execution_complete &= len(successful) == required and all(
            item["evidence"]["execution"]["status"] == "qualified" for item in successful
        )
        for point in cell["points"]:
            values = [item["evidence"]["points"][point["key"]]["wall_time_seconds"] for item in successful]
            mean = statistics.mean(values) if values else None
            stddev = statistics.stdev(values) if len(values) >= 2 else None
            cv = stddev / mean if stddev is not None else None
            inclusive = [point["source_wall_time_seconds"], *values]
            inclusive_stddev = statistics.stdev(inclusive) if values else None
            inclusive_cv = inclusive_stddev / statistics.mean(inclusive) if inclusive_stddev is not None else None
            points.append(
                {
                    "cell_id": cell["cell_id"],
                    "phase": cell["phase"],
                    "key": point["key"],
                    "coordinates": point["coordinates"],
                    "source_wall_time_seconds": point["source_wall_time_seconds"],
                    "samples_seconds": values,
                    "sample_count": len(values),
                    "mean_seconds": mean,
                    "minimum_seconds": min(values) if values else None,
                    "maximum_seconds": max(values) if values else None,
                    "sample_cv": cv,
                    "sample_stddev_seconds": stddev,
                    "source_inclusive_cv": inclusive_cv,
                    "source_inclusive_stddev_seconds": inclusive_stddev,
                    "source_inclusive_sample_count": len(inclusive),
                    "status": (
                        "incomplete"
                        if len(values) != required
                        else "passed"
                        if cv is not None
                        and cv <= frozen["policy"]["cv_threshold"]
                        and inclusive_cv is not None
                        and inclusive_cv <= frozen["policy"]["cv_threshold"]
                        else "unstable"
                    ),
                }
            )
    counts = Counter(item["status"] for item in points)
    repeatability = "unstable" if counts["unstable"] else "incomplete" if counts["incomplete"] else "passed"
    report.update(
        points=points,
        repeatability={
            "status": repeatability,
            "cv_threshold": frozen["policy"]["cv_threshold"],
            "point_counts": dict(counts),
            "measurement_count": "requested independent new measurements plus the original published measurement",
            "time_unit": "seconds",
            "standard_deviation_denominator": "n - 1",
        },
        execution={
            "status": "failed"
            if execution_failed or validation_failed
            else "qualified"
            if execution_complete
            else "incomplete"
        },
        status="failed"
        if execution_failed or validation_failed or repeatability == "unstable"
        else "passed"
        if repeatability == "passed" and execution_complete
        else "incomplete",
    )


def assess_repeatability(frozen_plan: dict[str, Any], report: dict[str, Any], *, cv_threshold: float) -> dict[str, Any]:
    """Reassess preserved samples without scheduling any additional collection.

    The returned assessment records the original frozen plan and the changed
    criterion. A caller must retain it at a new report path, preserving history.
    Source and raw sample evidence are revalidated before any reassessment.
    """
    if type(cv_threshold) not in (int, float) or not math.isfinite(cv_threshold) or cv_threshold < 0:
        raise ValueError("repeatability cv_threshold must be finite and nonnegative")
    if report.get("plan_sha256") != frozen_plan.get("sha256"):
        raise ValueError("repeatability report does not match its frozen plan")
    source = load_repeatability_source(frozen_plan["source_campaign_dir"])
    fresh = freeze_repeatability_plan(
        source,
        frozen_plan["source_campaign_dir"],
        frozen_plan["source_checkpoint"]["path"],
        **frozen_plan["policy"],
    )
    if fresh != frozen_plan:
        raise ValueError("repeatability source artifacts changed before reassessment")
    _validate_saved_samples(source, frozen_plan, report, Path(report["output_dir"]).resolve())
    result = copy.deepcopy(report)
    assessment_plan = copy.deepcopy(frozen_plan)
    assessment_plan["policy"]["cv_threshold"] = cv_threshold
    result["assessment"] = {"source_report_sha256": _canonical_hash(report), "cv_threshold": cv_threshold}
    _summarize(assessment_plan, result)
    return result


def _validate_saved_samples(source, frozen, report, root):
    if report.get("output_dir") != str(root):
        raise ValueError("repeatability report output directory changed")
    if report.get("policy") != frozen["policy"]:
        raise ValueError("repeatability saved measurement policy changed")
    expected_cells = {cell["cell_id"] for cell in frozen["cells"]}
    if set(report.get("samples", {})) != expected_cells:
        raise ValueError("repeatability saved sample cells do not match the plan")
    seen_attempts = {cell["source_attempt_id"] for cell in frozen["cells"]}
    for cell in frozen["cells"]:
        samples = report["samples"][cell["cell_id"]]
        if len(samples) != frozen["policy"]["samples"]:
            raise ValueError("repeatability saved sample count changed")
        plan = _subset_plan(source, cell)
        for index, sample in enumerate(samples, start=1):
            if sample.get("sample_index") != index or not isinstance(sample.get("attempts"), list):
                raise ValueError("repeatability saved sample indices are invalid")
            for attempt_index, attempt in enumerate(sample["attempts"], start=1):
                relative = Path("samples") / cell["cell_id"] / f"sample-{index:02d}" / f"attempt-{attempt_index:02d}"
                directory = root / relative
                if attempt.get("directory") != str(relative) or not directory.resolve().is_relative_to(root):
                    raise ValueError("repeatability saved attempt directory is invalid")
                if attempt.get("status") not in {"running", "interrupted", "failed", "passed"}:
                    raise ValueError("repeatability saved attempt status is invalid")
                if attempt.get("files") is not None and _file_manifest(directory) != attempt["files"]:
                    raise ValueError("repeatability attempt files changed")
                if attempt["status"] != "passed":
                    continue
                if attempt_index != len(sample["attempts"]):
                    raise ValueError("repeatability successful attempt was superseded")
                evidence = _sample_evidence(plan, cell, directory)
                if evidence != attempt.get("evidence"):
                    raise ValueError("saved repeatability sample evidence changed")
                if evidence["attempt_id"] in seen_attempts:
                    raise ValueError("repeatability reused an attempt instead of an independent launch")
                seen_attempts.add(evidence["attempt_id"])


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
) -> dict[str, Any]:
    """Remeasure only the frozen subset; retain separate raw data for every attempt.

    The caller owns execution authorization and an allocation for the selected
    Kubernetes or Slurm transport. No serving data or formal table is replaced.
    """
    if retry_failed and not resume:
        raise ValueError("repeatability retry_failed requires resume")
    if _canonical_hash(with_kv_warmup_defaults(generator_overrides)) != source_plan.generator_config_sha256:
        raise ValueError("repeatability deployment inputs differ from the original launch")
    frozen = freeze_repeatability_plan(
        source_plan,
        source_campaign_dir,
        source_checkpoint_path,
        samples=samples,
        max_points_per_cell=max_points_per_cell,
        cv_threshold=cv_threshold,
    )
    root = Path(output_dir).expanduser().resolve()
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
            "schema_version": 1,
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
        _atomic_json(root / REPORT_FILENAME, report)
    if any(cell["execution"]["status"] == "failed" for cell in frozen["cells"]):
        # A stable repeat set cannot qualify a measured wrong-source campaign.
        # Retain its assessment without spending another benchmark launch.
        _atomic_json(root / REPORT_FILENAME, report)
        return report
    for selected in frozen["cells"]:
        plan = _subset_plan(source_plan, selected)
        for sample in report["samples"][selected["cell_id"]]:
            previous = sample["attempts"][-1] if sample["attempts"] else None
            if previous is not None:
                previous_dir = root / previous["directory"]
                if previous["status"] == "passed":
                    continue
                if not retry_failed:
                    return report
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
                else:
                    validating = True
                    evidence = _sample_evidence(plan, selected, directory)
                    used_attempts = {cell["source_attempt_id"] for cell in frozen["cells"]}
                    used_attempts.update(
                        prior["evidence"]["attempt_id"]
                        for entries in report["samples"].values()
                        for entry in entries
                        for prior in entry["attempts"]
                        if prior.get("evidence")
                    )
                    if evidence["attempt_id"] in used_attempts:
                        raise ValueError("repeatability reused a collector attempt instead of an independent launch")
                    if evidence["execution"]["status"] == "failed":
                        attempt.update(
                            status="failed",
                            evidence=evidence,
                            failure_kind="validation_failed",
                            error="runtime execution evidence contradicts the frozen source",
                        )
                    else:
                        attempt.update(status="passed", evidence=evidence)
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
                _atomic_json(root / REPORT_FILENAME, report)
            if attempt["status"] != "passed":
                # Keep failures visible. Do not launch the remaining subset
                # repetitions into an uninvestigated runtime/cleanup failure.
                return report
    _summarize(frozen, report)
    _atomic_json(root / REPORT_FILENAME, report)
    return report
