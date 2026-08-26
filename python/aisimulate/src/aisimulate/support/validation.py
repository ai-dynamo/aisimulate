# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed MVP validation over matched E2E and held-out FPM evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .identity import support_cell_id
from .schema import (
    EvidenceBundle,
    EvidenceRecord,
    GateResult,
    SupportRequest,
    ValidationResult,
)

_ROLES = ("baseline", "top1", "top2", "top3")
_E2E_METRICS = ("ttft_ms", "tpot_ms", "output_throughput_tok_s")


def load_evidence(path: str | Path) -> EvidenceBundle:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read evidence {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed evidence YAML {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"evidence {source} must contain one mapping")
    return EvidenceBundle.model_validate(raw)


def _mape(records: list[EvidenceRecord]) -> float:
    return sum(abs(record.predicted - record.measured) / abs(record.measured) for record in records) / len(records)


def _database_gate(request: SupportRequest, systems_root: str | Path) -> GateResult:
    root = Path(systems_root)
    version_dir = root / "data" / request.identity.gpu / request.identity.framework / request.identity.framework_version
    parquet_path = version_dir / "fpm_forward_perf.parquet"
    metadata_path = version_dir / "fpm_forward_perf.metadata.json"
    required = (root / f"{request.identity.gpu}.yaml", parquet_path, metadata_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return GateResult(
            status="blocked",
            details={"systems_root": str(root), "version_dir": str(version_dir)},
            errors=[f"FPM publication is incomplete; missing {path}" for path in missing],
        )

    errors: list[str] = []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GateResult(
            status="failed",
            details={"metadata": str(metadata_path)},
            errors=[f"could not read FPM commit metadata: {exc}"],
        )
    expected = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "system": request.identity.gpu,
        "backend": request.identity.framework,
        "backend_version": request.identity.framework_version,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"FPM metadata {key}={metadata.get(key)!r}; expected {value!r}")
    model_paths = metadata.get("model_paths")
    if not isinstance(model_paths, list) or request.identity.model not in model_paths:
        errors.append(f"FPM metadata model_paths must contain the exact model {request.identity.model!r}")
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if metadata.get("parquet_sha256") != digest:
        errors.append("FPM parquet bytes do not match the commit metadata digest")
    row_count = metadata.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
        errors.append(f"FPM metadata row_count must be positive, got {row_count!r}")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(parquet_path)
        required_columns = {
            "cell_id",
            "model_path",
            "system",
            "backend",
            "backend_version",
            "workload_kind",
            "latency_ms",
            "source_plan_sha256",
            "collector_attempt_id",
            "runtime_run_id",
            "runtime_grid_digest",
        }
        missing_columns = sorted(required_columns - set(parquet.schema_arrow.names))
        if missing_columns:
            errors.append(f"FPM parquet is missing required columns {missing_columns}")
        if isinstance(row_count, int) and parquet.metadata.num_rows != row_count:
            errors.append(f"FPM parquet row count {parquet.metadata.num_rows} does not match metadata {row_count}")
    except Exception as exc:
        errors.append(f"could not validate FPM parquet: {type(exc).__name__}: {exc}")
    return GateResult(
        status="failed" if errors else "pass",
        details={
            "systems_root": str(root),
            "parquet": str(parquet_path),
            "metadata": str(metadata_path),
            "row_count": row_count,
            "parquet_sha256": digest,
        },
        errors=errors,
    )


def _contract_gate(request: SupportRequest, bundle: EvidenceBundle) -> GateResult:
    errors = []
    expected_cell = support_cell_id(request)
    if bundle.support_cell_id != expected_cell:
        errors.append(f"evidence support_cell_id {bundle.support_cell_id!r} != {expected_cell!r}")
    seen: set[tuple[Any, ...]] = set()
    for record in bundle.records:
        key = (
            record.phase,
            record.workload_id,
            record.config_role,
            record.candidate_id,
            record.metric,
        )
        if key in seen:
            errors.append(f"duplicate evidence record {key!r}")
        seen.add(key)
        if record.gpu_count != request.identity.gpu_count:
            errors.append(
                f"{record.source_run_id}: gpu_count={record.gpu_count} does not match cell budget "
                f"{request.identity.gpu_count}"
            )
    return GateResult(
        status="failed" if errors else "pass",
        details={"record_count": len(bundle.records), "expected_support_cell_id": expected_cell},
        errors=errors,
    )


def _e2e_gate(request: SupportRequest, records: list[EvidenceRecord]) -> tuple[GateResult, dict[tuple[str, str], str]]:
    errors: list[str] = []
    by_slot: dict[tuple[str, str], list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        if record.phase == "e2e" and record.workload_id is not None:
            by_slot[(record.workload_id, record.config_role)].append(record)

    candidate_by_slot: dict[tuple[str, str], str] = {}
    expected_slots = {(workload.id, role) for workload in request.workloads for role in _ROLES}
    for slot in sorted(expected_slots):
        slot_records = by_slot.get(slot, [])
        candidate_ids = {record.candidate_id for record in slot_records}
        metrics = {record.metric for record in slot_records}
        if len(candidate_ids) != 1:
            errors.append(f"{slot}: expected one candidate_id, got {sorted(candidate_ids)!r}")
        else:
            candidate_by_slot[slot] = next(iter(candidate_ids))
        missing = set(_E2E_METRICS) - metrics
        extra = metrics - set(_E2E_METRICS)
        if missing or extra:
            errors.append(f"{slot}: missing metrics={sorted(missing)}, extra metrics={sorted(extra)}")
        for metric in _E2E_METRICS:
            count = sum(record.metric == metric for record in slot_records)
            if count != 1:
                errors.append(f"{slot}: expected one {metric} record, got {count}")

    unexpected_slots = set(by_slot) - expected_slots
    if unexpected_slots:
        errors.append(f"unexpected E2E workload/role slots: {sorted(unexpected_slots)!r}")

    metric_results: dict[str, Any] = {}
    for metric in _E2E_METRICS:
        metric_records = [record for record in records if record.phase == "e2e" and record.metric == metric]
        if len(metric_records) != request.validation.min_e2e_pairs:
            errors.append(
                f"{metric}: expected {request.validation.min_e2e_pairs} matched E2E records, got {len(metric_records)}"
            )
            continue
        value = _mape(metric_records)
        metric_results[metric] = {
            "sample_count": len(metric_records),
            "mape": value,
            "max_mape": request.validation.max_mape,
        }
        if value > request.validation.max_mape:
            errors.append(f"{metric}: MAPE {value:.6f} exceeds {request.validation.max_mape:.6f}")

    return (
        GateResult(
            status="failed" if errors else "pass",
            details={"required_pair_count": len(expected_slots), "metrics": metric_results},
            errors=errors,
        ),
        candidate_by_slot,
    )


def _fpm_gate(
    request: SupportRequest,
    records: list[EvidenceRecord],
    candidate_by_slot: dict[tuple[str, str], str],
) -> GateResult:
    if not request.fpm.required:
        return GateResult(status="pass", details={"required": False})
    required_slot_count = len(request.workloads) * len(_ROLES)
    if len(candidate_by_slot) != required_slot_count:
        return GateResult(
            status="blocked",
            details={
                "required": True,
                "candidate_slot_count": len(candidate_by_slot),
                "required_candidate_slot_count": required_slot_count,
            },
            errors=[
                "held-out FPM accuracy requires complete E2E candidate mappings for "
                f"all {required_slot_count} workload/role slots"
            ],
        )
    errors: list[str] = []
    expected_candidates = set(candidate_by_slot.values())
    phase_results: dict[str, Any] = {}
    for phase in ("fpm_prefill", "fpm_decode"):
        phase_records = [record for record in records if record.phase == phase]
        by_candidate: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in phase_records:
            by_candidate[record.candidate_id].append(record)
        for candidate in sorted(expected_candidates):
            count = len(by_candidate.get(candidate, []))
            if count != 1:
                errors.append(f"{phase}: candidate {candidate!r} requires one held-out point, got {count}")
        extras = set(by_candidate) - expected_candidates
        if extras:
            errors.append(f"{phase}: unexpected candidates {sorted(extras)!r}")
        matched = [record for record in phase_records if record.candidate_id in expected_candidates]
        if len(matched) == len(expected_candidates) and matched:
            value = _mape(matched)
            phase_results[phase] = {
                "sample_count": len(matched),
                "mape": value,
                "max_mape": request.validation.max_mape,
            }
            if value > request.validation.max_mape:
                errors.append(f"{phase}: MAPE {value:.6f} exceeds {request.validation.max_mape:.6f}")
    return GateResult(
        status="failed" if errors else "pass",
        details={"required": True, "candidate_count": len(expected_candidates), "phases": phase_results},
        errors=errors,
    )


def _value_gate(request: SupportRequest, records: list[EvidenceRecord]) -> GateResult:
    threshold = request.validation.recommendation_uplift_min
    if threshold is None:
        return GateResult(
            status="blocked",
            details={"threshold": None},
            errors=["validation.recommendation_uplift_min requires product sign-off before support can pass"],
        )
    errors: list[str] = []
    workload_results: dict[str, Any] = {}
    for workload in request.workloads:
        throughput = [
            record
            for record in records
            if record.phase == "e2e"
            and record.workload_id == workload.id
            and record.metric == "output_throughput_tok_s"
        ]
        by_role = {record.config_role: record for record in throughput}
        if "baseline" not in by_role or "top1" not in by_role:
            errors.append(f"{workload.id}: baseline and top1 throughput evidence are required")
            continue
        baseline = by_role["baseline"]
        top1 = by_role["top1"]
        if not baseline.slo_compliant or not top1.slo_compliant:
            errors.append(f"{workload.id}: baseline and top1 must both be SLO-compliant")
        uplift = top1.measured / baseline.measured
        workload_results[workload.id] = {
            "baseline": baseline.measured,
            "top1": top1.measured,
            "uplift": uplift,
            "threshold": threshold,
        }
        if uplift < threshold:
            errors.append(f"{workload.id}: uplift {uplift:.6f} is below {threshold:.6f}")
    return GateResult(
        status="failed" if errors else "pass",
        details={"workloads": workload_results},
        errors=errors,
    )


def validate_evidence(
    request: SupportRequest,
    bundle: EvidenceBundle,
    *,
    systems_root: str | Path,
) -> ValidationResult:
    contract = _contract_gate(request, bundle)
    database = _database_gate(request, systems_root)
    e2e, candidate_by_slot = _e2e_gate(request, bundle.records)
    fpm = _fpm_gate(request, bundle.records, candidate_by_slot)
    value = _value_gate(request, bundle.records)
    gates = {
        "contract_completeness": contract,
        "fpm_publication": database,
        "e2e_accuracy": e2e,
        "fpm_accuracy": fpm,
        "recommendation_value": value,
    }
    errors = [error for gate in gates.values() for error in gate.errors]
    if any(gate.status == "failed" for gate in gates.values()):
        status = "failed"
    elif any(gate.status == "blocked" for gate in gates.values()):
        status = "blocked"
    else:
        status = "pass"
    return ValidationResult(
        support_cell_id=support_cell_id(request),
        status=status,
        gates=gates,
        errors=errors,
    )
