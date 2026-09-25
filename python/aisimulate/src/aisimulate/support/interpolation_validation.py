# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate direct FPM interpolation against withheld measured coordinates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import yaml

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

from .schema import SupportRequest

_PHASES = ("prefill", "decode")
_COORDINATES = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def _config(request: SupportRequest, root: Path) -> ForwardPassPerfModelConfig:
    deployment = request.profile_deployment()
    if deployment is None:
        raise ValueError("interpolation validation requires an explicit FPM model profile")
    return ForwardPassPerfModelConfig(
        model=request.identity.model,
        system=request.identity.gpu,
        backend=request.identity.framework,
        backend_version=request.identity.framework_version,
        worker_type="aggregated",
        tp=deployment.tp,
        pp=deployment.pp,
        attention_dp=deployment.dp,
        moe_tp_size=deployment.moe_tp,
        moe_ep_size=deployment.moe_ep,
        fpm_profile=request.fpm_profile.model_dump(mode="json"),
        systems_paths=(str(root),),
        estimation_mode="fpm_interpolation",
        fallback_policy="deny",
        estimator_config={"fpm_interpolation": {"method": "direct", "collect_coverage": True}},
    )


def _matches(row: dict[str, Any], request: SupportRequest) -> bool:
    deployment = request.profile_deployment().model_dump(mode="json", exclude={"resources"})
    return (
        row["model_path"] == request.identity.model
        and all(row.get(name) == value for name, value in deployment.items())
        and all(
            row.get(name, default) == default
            for name, default in zip(EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY, strict=True)
        )
    )


def _coordinate(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (row["workload_kind"], *(row[name] for name in _COORDINATES))


def _point(coordinate: tuple[str, int, int, int]) -> dict[str, Any]:
    return {"phase": coordinate[0], **dict(zip(_COORDINATES, coordinate[1:], strict=True))}


def _execution_boundaries(
    rows: list[dict[str, Any]], evidence: list[dict[str, Any]] | None
) -> tuple[set[tuple[str, int, int, int]], dict[str, Any]]:
    """Retain adjacent measured mode transitions, without predicting dispatch or timing."""
    report: dict[str, Any] = {
        "status": "not_assessed",
        "basis": "native_expected_point_modes_and_resolved_graph_configuration",
        "per_point_observed_dispatch": "unreported",
        "cells": [],
        "transitions": [],
        "retained_anchors": [],
        "additional_to_envelope": [],
        "issues": [],
    }
    if evidence is None:
        report["issues"].append("formal table coordinates alone do not establish CUDA graph boundaries")
        return set(), report
    if {cell["cell_id"] for cell in evidence} != {row["cell_id"] for row in rows} or len(evidence) != len(
        {cell["cell_id"] for cell in evidence}
    ):
        raise ValueError("holdout execution evidence must match the formal deployment cells exactly")
    modes = {}
    for cell in sorted(evidence, key=lambda item: item["cell_id"]):
        cell_id = cell["cell_id"]
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        identity = cell["identity"]
        if set(identity) != {
            "source_plan_sha256",
            "collector_attempt_id",
            "runtime_run_id",
            "runtime_grid_digest",
        } or any(any(not value or row.get(name) != value for name, value in identity.items()) for row in cell_rows):
            raise ValueError(f"holdout execution evidence has a different source identity: {cell_id}")
        points = {_coordinate(point): point for point in cell["points"]}
        if len(points) != len(cell["points"]) or set(points) != {_coordinate(row) for row in cell_rows}:
            raise ValueError(f"holdout execution evidence coordinates differ from the formal cell: {cell_id}")
        graphs = sorted(cell["native_graph_config"], key=lambda graph: graph["dp_rank"])
        ranks = [graph["dp_rank"] for graph in graphs]
        if len(ranks) != len(set(ranks)) or set(ranks) != set(range(cell_rows[0]["dp"])):
            raise ValueError(f"holdout graph evidence has inconsistent DP ranks: {cell_id}")
        for graph in graphs:
            if _identity(Path(graph["source"]["path"])) != graph["source"]:
                raise ValueError(f"holdout native graph source changed: {cell_id}")
        configs = [graph["config"] for graph in graphs]
        present = [config for config in configs if isinstance(config, dict)]
        if any(config != present[0] for config in present):
            raise ValueError(f"holdout native graph configurations disagree across ranks: {cell_id}")
        config = present[0] if len(present) == len(configs) else {}
        phase = cell_rows[0]["workload_kind"]
        captures = config.get(f"{phase}_capture_sizes", config.get("capture_sizes"))
        configured_mode = config.get(f"{phase}_mode", config.get("mode"))
        configured_modes = {
            "NONE": {"NONE"},
            "FULL": {"NONE", "FULL"},
            "PIECEWISE": {"NONE", "PIECEWISE"},
            "FULL_AND_PIECEWISE": {"NONE", "PIECEWISE" if phase == "prefill" else "FULL"},
            "FULL_DECODE_ONLY": {"NONE"} if phase == "prefill" else {"NONE", "FULL"},
        }.get(configured_mode if isinstance(configured_mode, str) else None)
        if (
            configured_modes is None
            or not isinstance(captures, list)
            or any(type(size) is not int or size < 1 for size in captures)
        ):
            report["issues"].append(f"{cell_id}: resolved native graph configuration is missing or unknown")
            configured_modes = None
        maximum = config.get("max_capture_size")
        if (
            configured_modes is not None
            and maximum is not None
            and (type(maximum) is not int or maximum < max(captures, default=0))
        ):
            raise ValueError(f"holdout native capture sizes exceed the reported maximum: {cell_id}")
        known = 0
        for row in cell_rows:
            if row.get("kv_seed_regime") == "fake_fallback":
                continue
            coordinate = _coordinate(row)
            point = points[coordinate]
            mode, capture = point.get("expected_cudagraph_mode"), point.get("expected_capture_size")
            if configured_modes is None or not isinstance(mode, str) or mode not in {"NONE", "FULL", "PIECEWISE"}:
                continue
            if mode != "NONE" and capture is None:
                continue
            scheduled = row["total_prefill_tokens"] if phase == "prefill" else row["batch_size"]
            if mode not in configured_modes or (
                capture is not None
                if mode == "NONE"
                else type(capture) is not int or capture < scheduled or capture not in captures
            ):
                raise ValueError(f"holdout point mode/capture contradicts native graph configuration: {cell_id}")
            modes[coordinate] = mode
            known += 1
        eligible = sum(row.get("kv_seed_regime") != "fake_fallback" for row in cell_rows)
        if known < eligible:
            report["issues"].append(f"{cell_id}: {eligible - known} eligible points lack usable expected-mode evidence")
        report["cells"].append(
            {
                "cell_id": cell_id,
                "identity": identity,
                "native_graph_config": graphs,
                "eligible_point_count": eligible,
                "known_mode_point_count": known,
            }
        )
    anchors = set()
    eligible_rows = [row for row in rows if row.get("kv_seed_regime") != "fake_fallback"]
    for axis in _COORDINATES:
        fixed = [name for name in _COORDINATES if name != axis]
        curves: dict[tuple, list[dict[str, Any]]] = {}
        for row in eligible_rows:
            curves.setdefault((row["workload_kind"], *(row[name] for name in fixed)), []).append(row)
        for _key, curve in sorted(curves.items()):
            ordered = sorted(curve, key=lambda row: row[axis])
            for lower, upper in itertools.pairwise(ordered):
                lo, hi = _coordinate(lower), _coordinate(upper)
                if lo in modes and hi in modes and modes[lo] != modes[hi]:
                    anchors.update((lo, hi))
                    report["transitions"].append(
                        {
                            "axis": axis,
                            "lower": {**_point(lo), "mode": modes[lo]},
                            "upper": {**_point(hi), "mode": modes[hi]},
                        }
                    )
    envelope = set().union(
        *(_anchors([row for row in eligible_rows if row["workload_kind"] == phase]) for phase in _PHASES)
    )
    report.update(
        status="incomplete" if report["issues"] else "assessed",
        retained_anchors=[_point(coordinate) for coordinate in sorted(anchors)],
        additional_to_envelope=[_point(coordinate) for coordinate in sorted(anchors - envelope)],
    )
    return anchors, report


def _anchors(rows: list[dict[str, Any]]) -> set[tuple[str, int, int, int]]:
    """Retain the envelope at every batch, including both outer KV curves."""
    anchors = set()
    for batch in sorted({row["batch_size"] for row in rows}):
        group = [row for row in rows if row["batch_size"] == batch]
        if group[0]["workload_kind"] == "decode":
            anchors.update(_coordinate(fn(group, key=_coordinate)) for fn in (min, max))
            continue
        kv_values = {row["total_kv_read_tokens"] for row in group}
        for kv in (min(kv_values), max(kv_values)):
            curve = [row for row in group if row["total_kv_read_tokens"] == kv]
            anchors.update(_coordinate(fn(curve, key=_coordinate)) for fn in (min, max))
        # A ragged interior KV curve may extend beyond both outer curves.
        # Preserve those token-envelope extrema as well.
        anchors.update(_coordinate(fn(group, key=_coordinate)) for fn in (min, max))
    return anchors


def _select(
    rows: list[dict[str, Any]], limit: int, seed: int, execution_anchors: set[tuple[str, int, int, int]]
) -> list[dict[str, Any]]:
    """Target 20% up to the cap, retaining anchors and spreading over log-scaled shapes."""
    if len(rows) < 2:
        return []
    anchors = _anchors(rows) | execution_anchors
    ordered = sorted((row for row in rows if _coordinate(row) not in anchors), key=_coordinate)
    if not ordered:
        return []
    count = min(limit, max(1, len(rows) // 5), len(ordered))
    values = [[math.log2(1 + row[name]) for name in _COORDINATES] for row in ordered]
    bounds = [(min(axis), max(axis)) for axis in zip(*values, strict=True)]
    positions = [
        [(value - lo) / (hi - lo) if hi > lo else 0.0 for value, (lo, hi) in zip(vector, bounds, strict=True)]
        for vector in values
    ]

    def tie_break(index: int) -> str:
        return hashlib.sha256(_json_bytes([seed, _coordinate(ordered[index])])).hexdigest()

    selected = [min(range(len(ordered)), key=tie_break)]
    remaining = set(range(len(ordered))) - set(selected)
    while len(selected) < count:
        index = max(
            remaining,
            key=lambda candidate: (
                min(
                    sum((a - b) ** 2 for a, b in zip(positions[candidate], positions[chosen], strict=True))
                    for chosen in selected
                ),
                tie_break(candidate),
            ),
        )
        selected.append(index)
        remaining.remove(index)
    return [ordered[index] for index in selected]


def _boundary_axes(row: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    axes = []
    for name in _COORDINATES:
        values = [point[name] for point in rows]
        lo, hi = min(values), max(values)
        if lo != hi:
            if row[name] == lo:
                axes.append(f"min_{name}")
            if row[name] == hi:
                axes.append(f"max_{name}")
    return axes


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    if row["workload_kind"] == "prefill":
        scheduled = {
            "num_prefill_requests": row["batch_size"],
            "sum_prefill_tokens": row["total_prefill_tokens"],
            "sum_prefill_kv_tokens": row["total_kv_read_tokens"],
        }
    else:
        scheduled = {"num_decode_requests": row["batch_size"], "sum_decode_kv_tokens": row["total_kv_read_tokens"]}
    return {"scheduled_requests": scheduled}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _phase_report(
    rows: list[dict[str, Any]], *, max_p95_relative_error: float, max_unsupported_fraction: float
) -> dict[str, Any]:
    errors = [row["relative_error"] for row in rows if row["status"] == "predicted"]
    unsupported = sum(row["status"] == "unsupported" for row in rows)
    fraction = unsupported / len(rows) if rows else None
    absolute = [abs(error) for error in errors]
    distribution = (
        {
            "p50": _percentile(absolute, 0.5),
            "p95": _percentile(absolute, 0.95),
            "max": max(absolute),
            "bias": sum(errors) / len(errors),
        }
        if errors
        else None
    )
    issues = []
    if not rows:
        issues.append("no eligible coordinates remain after retaining measured envelope and execution-boundary anchors")
    elif not errors:
        issues.append("no withheld coordinate has a supported interpolation prediction")
    if fraction is not None and fraction > max_unsupported_fraction:
        issues.append("unsupported query fraction exceeds the configured threshold")
    if distribution is not None and distribution["p95"] > max_p95_relative_error:
        issues.append("p95 absolute relative error exceeds the configured threshold")
    return {
        "status": "passed" if not issues else ("failed" if rows else "not_assessed"),
        "selected_count": len(rows),
        "predicted_count": len(errors),
        "unsupported_count": unsupported,
        "unsupported_fraction": fraction,
        "boundary_unsupported_count": sum(
            row["status"] == "unsupported" and bool(row["boundary_axes"]) for row in rows
        ),
        "absolute_relative_error": distribution,
        "bias_definition": "mean((predicted_ms - measured_ms) / measured_ms)",
        "issues": issues,
    }


def evaluate_interpolation_holdout(
    request: SupportRequest,
    *,
    systems_root: str | Path,
    output_dir: str | Path,
    max_points_per_phase: int = 16,
    seed: int = 42,
    max_p95_relative_error: float = 0.20,
    max_unsupported_fraction: float = 0.0,
    execution_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a reproducible native direct-interpolation assessment to a fresh directory.

    Forward timing permits pending memory. This tests interpolation, not request
    admission, measurement repeatability or serving accuracy. The measured
    envelope and evidenced execution-mode boundaries are retained; any missing
    interior brackets remain unsupported. Changing policy or inputs requires
    another directory; source measurements are never edited.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(max_points_per_phase) is not int or max_points_per_phase < 1:
        raise ValueError("max_points_per_phase must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value in (
        ("max_p95_relative_error", max_p95_relative_error),
        ("max_unsupported_fraction", max_unsupported_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if max_unsupported_fraction > 1:
        raise ValueError("max_unsupported_fraction must be at most 1")
    source = Path(systems_root).expanduser().resolve()
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink():
        raise ValueError("holdout output must not be a symbolic link")
    output = output.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("holdout output must be separate from the source systems directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("holdout output must be a fresh or empty directory")
    deployment = request.profile_deployment()
    if deployment is None:
        raise ValueError("interpolation validation requires an explicit FPM model profile")
    relative = Path("data") / deployment.system / deployment.backend / deployment.backend_version
    parquet_relative = relative / "fpm_forward_perf.parquet"
    metadata_relative = parquet_relative.with_suffix(".metadata.json")
    system_relative = Path(f"{deployment.system}.yaml")
    source_paths = [source / relative for relative in (system_relative, parquet_relative, metadata_relative)]
    if any(not path.is_file() or not path.resolve().is_relative_to(source) for path in source_paths):
        raise ValueError("source systems directory must contain a complete local system YAML and FPM data pair")
    source_bytes = [path.read_bytes() for path in source_paths]
    inputs = {
        name: {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        for name, path, content in zip(("system", "parquet", "metadata"), source_paths, source_bytes, strict=True)
    }
    system = yaml.safe_load(source_bytes[0])
    if not isinstance(system, dict) or system.get("data_dir") != f"data/{deployment.system}":
        raise ValueError("holdout validation requires a system YAML with local data_dir=data/<system>")
    table = pq.read_table(pa.BufferReader(source_bytes[1]))
    metadata = json.loads(source_bytes[2])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source-systems"
    for relative_path, content in zip(
        (system_relative, parquet_relative, metadata_relative), source_bytes, strict=True
    ):
        destination = snapshot / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    # Native construction checks the full source pair, exact identity, schema,
    # digest, duplicate physical keys and coordinate collisions before selection.
    baseline = RustForwardPassPerfModel.best_available(_config(request, snapshot))
    baseline.close()
    all_rows = table.to_pylist()
    matching = [row for row in all_rows if _matches(row, request)]
    if not matching:
        raise ValueError("FPM data contains no exact profile deployment cell")
    eligible = [row for row in matching if row.get("kv_seed_regime") != "fake_fallback"]
    execution_anchors, boundaries = _execution_boundaries(matching, execution_evidence)
    selected = []
    for phase in _PHASES:
        phase_rows = [row for row in eligible if row["workload_kind"] == phase]
        selected.extend(
            {**row, "boundary_axes": _boundary_axes(row, phase_rows)}
            for row in _select(phase_rows, max_points_per_phase, seed, execution_anchors)
        )
    coordinates = {_coordinate(row) for row in selected}
    # Exclude every physical row at each selected cell-coordinate, independently
    # of collector cell ID, sample ID or DP-rank/provenance fields.
    retained_indices = [
        index for index, row in enumerate(all_rows) if not (_matches(row, request) and _coordinate(row) in coordinates)
    ]
    plan = {
        "schema_version": 1,
        "scope": "withheld_coordinate_direct_interpolation",
        "inputs": inputs,
        "request_sha256": hashlib.sha256(_json_bytes(request.model_dump(mode="json"))).hexdigest(),
        "policy": {
            "seed": seed,
            "max_points_per_phase": max_points_per_phase,
            "target_holdout_fraction": 0.20,
            "small_table_minimum_points": 1,
            "max_p95_relative_error": max_p95_relative_error,
            "max_unsupported_fraction": max_unsupported_fraction,
            "selection": "retained_envelope_execution_boundaries_seeded_farthest_shape_v2",
            "fold": "simultaneously_remove_all_selected_coordinates",
        },
        "capture_boundaries": boundaries,
        "source_row_count": len(all_rows),
        "matching_row_count": len(matching),
        "eligible_row_count": len(eligible),
        "excluded_fake_fallback_count": len(matching) - len(eligible),
        "removed_row_count": len(all_rows) - len(retained_indices),
        "training_row_count": len(retained_indices),
        "retained_envelope_anchors": [
            _point(coordinate)
            for phase in _PHASES
            for coordinate in sorted(_anchors([row for row in eligible if row["workload_kind"] == phase]))
        ],
        "selected_points": selected,
    }
    plan_path = output / "holdout-plan.json"
    plan_path.write_bytes(_json_bytes(plan))
    training = output / "systems"
    (training / system_relative).parent.mkdir(parents=True, exist_ok=True)
    (training / system_relative).write_bytes(source_bytes[0])
    parquet = training / parquet_relative
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(retained_indices, type=pa.int64())), parquet, compression="zstd")
    metadata.update(
        row_count=len(retained_indices),
        parquet_sha256=_identity(parquet)["sha256"],
        validation_derivation={"kind": "coordinate_holdout", "plan": _identity(plan_path)},
    )
    (training / metadata_relative).write_bytes(_json_bytes(metadata))
    config = _config(request, training)
    config_path = output / "perfmodel-config.json"
    config_path.write_bytes(_json_bytes(config.to_dict()))
    model = RustForwardPassPerfModel.best_available(config)
    predictions = []
    try:
        diagnostics = model.diagnostics()
        for row in selected:
            point = {
                "phase": row["workload_kind"],
                "coordinates": {name: row[name] for name in _COORDINATES},
                "cell_id": row["cell_id"],
                "kv_seed_regime": row.get("kv_seed_regime"),
                "boundary_axes": row["boundary_axes"],
                "measured_ms": row["latency_ms"],
            }
            try:
                predicted = model.estimate_forward_pass_time_ms(_metrics(row))
            except PerfDataNotAvailableError as error:
                point.update(status="unsupported", reason=str(error))
            else:
                if predicted is None or not math.isfinite(predicted) or predicted <= 0:
                    raise ValueError("native interpolation returned a non-positive or non-finite timing")
                point.update(
                    status="predicted",
                    predicted_ms=predicted,
                    relative_error=(predicted - row["latency_ms"]) / row["latency_ms"],
                )
            predictions.append(point)
        coverage = model.fpm_query_coverage()
    finally:
        model.close()
    if not coverage or coverage["queries"]["measured"]:
        raise ValueError("holdout evaluation encountered a measured lookup; withheld data leaked into training")
    phases = {
        phase: _phase_report(
            [row for row in predictions if row["phase"] == phase],
            max_p95_relative_error=max_p95_relative_error,
            max_unsupported_fraction=max_unsupported_fraction,
        )
        for phase in _PHASES
    }
    unchanged = all(_identity(Path(identity["path"])) == identity for identity in inputs.values())
    unchanged &= all(
        _identity(Path(graph["source"]["path"])) == graph["source"]
        for cell in boundaries["cells"]
        for graph in cell["native_graph_config"]
    )
    status = "passed"
    if not unchanged or any(item["status"] == "failed" for item in phases.values()):
        status = "failed"
    elif boundaries["status"] == "incomplete" or any(item["status"] == "not_assessed" for item in phases.values()):
        status = "incomplete"
    report = {
        "schema_version": 1,
        "status": status,
        "scope": "forward_timing_interpolation_only",
        "serving_accuracy": "not_assessed",
        "measurement_noise": "not_separated_from_interpolation_error",
        "reference_measurement_policy": metadata["measurement_policy"],
        "limitations": [
            "Each reference is the original single-sample formal-table measurement; separate subset "
            "repeatability evidence does not remove measurement noise from these errors.",
            "The bounded representative holdout is not a statistical confidence bound or full-domain accuracy claim.",
            "When available, boundary retention uses native expected point modes and resolved graph configuration, "
            "not observed per-call dispatch.",
            "This onboarding selection retains evidenced mode boundaries; it does not test missing-boundary "
            "interpolation or change predict/recommend.",
            "Capture-bucket and kernel changes within one graph mode are not classified by this assessment.",
        ],
        "source_artifacts_unchanged": unchanged,
        "policy": plan["policy"],
        "capture_boundaries": boundaries,
        "phases": phases,
        "predictions": predictions,
        "native_query_coverage": coverage,
        "native_diagnostics": diagnostics,
        "artifacts": {
            "holdout_plan": _identity(plan_path),
            "perfmodel_config": _identity(config_path),
            "training_parquet": _identity(parquet),
            "training_metadata": _identity(training / metadata_relative),
            "training_system": _identity(training / system_relative),
            "source": inputs,
        },
        "issues": [] if unchanged else ["source FPM artifacts changed during validation"],
    }
    (output / "interpolation-validation.json").write_bytes(_json_bytes(report))
    return report
