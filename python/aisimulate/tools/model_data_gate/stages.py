# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable phase-one diagnostics; deliberately do not impersonate native probes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "tools/perf_database"))

import check_cross_backend as cross
import parquet_diff
from collector.op_catalog import load_family_map
from collector.provenance import case_plan_hash, validate_collection_meta_for_update
from power_data import power_metric_issues
from tools.model_data_gate.run import APP, DATA, SYSTEMS, result

# Physical GEMM shape pinned independently by the native loader in
# crates/core/src/perfmodel/perf_database/gemm.rs. Other op schemas remain
# explicitly incomplete instead of being inferred from whatever file arrived.
GEMM_REQUIRED = {"framework", "version", "device", "op_name", "kernel_source", "gemm_dtype", "m", "n", "k", "latency"}
METADATA = {"framework", "version", "device", "op_name"}
FRAMEWORK_ALIASES = {"tensorrtllm": "trtllm", "trtllm": "trtllm", "sglang": "sglang", "vllm": "vllm"}


def framework(value: object) -> str:
    name = str(value).lower().replace("-", "").replace("_", "")
    return FRAMEWORK_ALIASES.get(name, name)


def selected_tables(roots: dict[str, Path], paths: list[str]) -> list[str]:
    tables = {path for path in paths if path.startswith(DATA) and path.endswith(".parquet")}
    for path in paths:
        # A sidecar or reuse change rechecks all tables in that version. A
        # systems spec change affects all families/backends under that system.
        if path.startswith(DATA) and Path(path).name in {"collection_meta.yaml", "reuse.yaml"}:
            prefix = str(Path(path).parent)
        elif path.startswith(SYSTEMS + "/") and Path(path).parent.as_posix() == SYSTEMS and path.endswith(".yaml"):
            prefix = DATA + Path(path).stem
        else:
            continue
        for root in roots.values():
            tables.update(p.relative_to(root).as_posix() for p in (root / prefix).rglob("*.parquet"))
    return sorted(tables)


def finding(path: str, rule: str, observed, expected: str, *, row: int | None = None) -> dict:
    return {
        "file": path,
        "row": row,
        "rule": rule,
        "observed": observed,
        "expected": expected,
        "remediation": "Correct the artifact or attested metadata; retain original measurement evidence.",
    }


def read_table(path: Path):
    if path.read_bytes().startswith(parquet_diff.LFS_POINTER_PREFIX):
        raise ValueError("unresolved Git LFS pointer; real Parquet bytes are required")
    # ParquetFile avoids dataset/hive partition inference from parent directory names.
    return pq.ParquetFile(path).read()


def validate_table(path: str, table, family_map: dict) -> tuple[list[dict], list[str]]:
    failures, gaps = [], []
    parts = path.removeprefix(DATA).split("/")
    if len(parts) != 5:
        return [finding(path, "directory_identity", parts, "system/family/backend/version/table.parquet")], []
    _system, family, backend, version, filename = parts
    stem = Path(filename).stem
    if family_map.get(stem) != family:
        failures.append(finding(path, "family_identity", family, f"catalog family {family_map.get(stem)!r}"))
    names = table.column_names
    if len(set(names)) != len(names):
        return [finding(path, "duplicate_columns", names, "unique column names")], []
    if table.num_rows == 0:
        failures.append(finding(path, "empty_table", 0, "at least one measurement"))
    if filename == "gemm_perf.parquet":
        missing = sorted(GEMM_REQUIRED - set(names))
        if missing:
            failures.append(finding(path, "required_columns", missing, "complete GEMM schema"))
        for key in ("m", "n", "k"):
            if key in names and not pa.types.is_integer(table.schema.field(key).type):
                failures.append(
                    finding(path, "shape_type", f"{key}: {table.schema.field(key).type}", "integer dimensions")
                )
    else:
        gaps.append(f"{path}: operation-specific schema/type/physical-key contract is not integrated")
    timing = [name for name in names if name in {"latency", "avg_ms", "dispatch_avg_t_us", "combine_avg_t_us"}]
    if not timing:
        failures.append(finding(path, "timing_schema", names, "recognized latency column(s)"))
    for name in timing:
        if not (
            pa.types.is_floating(table.schema.field(name).type) or pa.types.is_integer(table.schema.field(name).type)
        ):
            failures.append(finding(path, "timing_type", f"{name}: {table.schema.field(name).type}", "numeric timing"))
    # Metadata is checked separately and must not make duplicate physical keys
    # appear unique. Keep kernel_source because kernels may measure the same shape.
    keys = (
        [name for name in ("kernel_source", "gemm_dtype", "m", "n", "k") if name in names]
        if filename == "gemm_perf.parquet"
        else [name for name in names if name not in METADATA | parquet_diff.MEASUREMENT_COLUMNS]
    )
    seen = set()
    for index, row in enumerate(table.to_pylist()):
        for name, value in row.items():
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                failures.append(
                    finding(path, "finite_nonnull", f"{name}={value!r}", "non-null finite value", row=index)
                )
        for name in timing:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue  # schema/finite rules carry the failure
            # Only computescale is a signed difference. Raw score-calibration
            # durations may be zero but may never be negative.
            minimum = -math.inf if filename == "computescale_perf.parquet" else 0
            allow_zero = filename in {
                "computescale_perf.parquet",
                "dsv4_csa_topk_calib_perf.parquet",
                "glm5_topk_module_perf.parquet",
            }
            if value < minimum or (value == 0 and not allow_zero):
                failures.append(
                    finding(
                        path,
                        "timing_boundary",
                        {name: value},
                        "finite signed computescale delta, nonnegative calibration, otherwise positive",
                        row=index,
                    )
                )
        for name, expected in (("framework", backend), ("version", version)):
            if name in row:
                actual = framework(row[name]) if name == "framework" else str(row[name])
                if actual != expected:
                    failures.append(finding(path, "row_identity", {name: actual}, expected, row=index))
        if filename == "gemm_perf.parquet":
            if "op_name" in row and row["op_name"] != "gemm":
                failures.append(finding(path, "operation_identity", row["op_name"], "gemm", row=index))
            for name in ("m", "n", "k"):
                value = row.get(name)
                if isinstance(value, (int, float)) and value <= 0:
                    failures.append(finding(path, "shape_boundary", {name: value}, "positive dimension", row=index))
        key = tuple(parquet_diff._freeze_value(row[name]) for name in keys)
        if key in seen:
            failures.append(
                finding(
                    path,
                    "duplicate_physical_key",
                    {name: repr(row[name]) for name in keys},
                    "unique shape/kernel coordinate",
                    row=index,
                )
            )
        seen.add(key)
    failures.extend(
        finding(path, "power_schema", issue, "paired float64 power fields or explicit zero sentinel")
        for issue in power_metric_issues(table)
    )
    gaps.append(f"{path}: device/topology identity and production source classification require native probes")
    return failures, gaps


def validate_metadata(root: Path, path: str, table) -> tuple[list[dict], list[str]]:
    sidecar = (root / path).parent / "collection_meta.yaml"
    failures, gaps = [], []
    try:
        metadata = yaml.safe_load(sidecar.read_text())
        validate_collection_meta_for_update(metadata)
        if metadata.get("provenance") not in {None, "collected", "local"}:
            raise ValueError("table provenance must describe a collection, not estimated or synthetic data")
        backend, version = Path(path).parts[-3:-1]
        runtime = metadata["runtime"]
        if framework(runtime["framework"]) != backend or str(runtime["version"]) != version:
            failures.append(finding(path, "sidecar_identity", runtime, f"{backend}/{version}"))
        entry = metadata["tables"][Path(path).stem]
        if "rows" in entry and entry["rows"] != table.num_rows:
            failures.append(finding(path, "sidecar_rows", entry["rows"], str(table.num_rows)))
        events = entry.get("collections", [entry])
        for event in events:
            if event.get("case_plan_hash") == case_plan_hash([]):
                failures.append(
                    finding(path, "empty_case_plan", event["case_plan_hash"], "attested nonempty attempted-case plan")
                )
            if "case_plan_hash" not in event:
                gaps.append(f"{path}: reduced historical sidecar is not full measurement attestation")
    except Exception as error:
        failures.append(
            finding(
                path, "collection_provenance", f"{type(error).__name__}: {error}", "valid applicable Collector sidecar"
            )
        )
    return failures, gaps


def artifact_integrity(roots: dict[str, Path], paths: list[str], out: Path) -> dict:
    catalog = roots["head"] / APP / "collector/op_backend_catalog.yaml"
    family_map = load_family_map(catalog)
    if not family_map:
        raise ValueError("head operation catalog is missing or empty")
    selected = selected_tables(roots, paths)
    failures, gaps, artifacts = [], [], []
    for path in selected:
        snapshots = {}
        artifact = {"file": path}
        for side, root in roots.items():
            source = root / path
            if not source.exists():
                snapshots[side] = None
                continue
            try:
                table = read_table(source)
                snapshots[side] = parquet_diff.Snapshot(path, table)
                artifact[side] = {
                    "rows": table.num_rows,
                    "schema": snapshots[side].schema,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
                if side == "head":
                    issues, missing = validate_table(path, table, family_map)
                    failures.extend(issues)
                    gaps.extend(missing)
                    issues, missing = validate_metadata(root, path, table)
                    failures.extend(issues)
                    gaps.extend(missing)
            except Exception as error:
                snapshots[side] = None
                failures.append(
                    finding(
                        path,
                        "readable_artifact",
                        f"{side}: {type(error).__name__}: {error}",
                        "readable Parquet artifact",
                    )
                )
        if snapshots.get("base") is not None or snapshots.get("head") is not None:
            artifact["row_diff"] = dataclasses.asdict(
                parquet_diff._diff_snapshots(path, snapshots["base"], snapshots["head"], detail_dir=out / "row-diffs")
            )
        if not (roots["head"] / path).exists():
            gaps.append(f"{path}: deletion requires production coverage and behavior evidence")
        artifacts.append(artifact)
    if not selected:
        gaps.append(
            "No directly changed tables; indirect model/collector/resolver/tooling changes require expanded coverage"
        )
    return result(
        "FAIL" if failures else "INCOMPLETE" if gaps else "PASS",
        f"Validated {len(selected)} changed/metadata-affected tables; {len(failures)} artifact violations",
        findings=failures,
        coverage_gaps=sorted(set(gaps)),
        artifacts=artifacts,
    )


def numerical_sanity(roots: dict[str, Path], paths: list[str]) -> dict:
    selected = selected_tables(roots, paths)
    findings, gaps, comparisons = [], [], []
    for path in selected:
        parts = path.removeprefix(DATA).split("/")
        if len(parts) != 5:
            gaps.append(f"{path}: invalid layout")
            continue
        system, _family, backend, version, filename = parts
        by_side = {}
        for side, root in roots.items():
            target = root / path
            if not target.exists():
                by_side[side] = []
                continue
            group = {}
            # The general checker examines only latest-per-backend. Override
            # the changed backend with THIS exact version, including old releases.
            for candidate in (root / DATA / system).glob(f"*/*/*/{filename}"):
                peer_backend, peer_version = candidate.parts[-3:-1]
                if peer_backend != backend:
                    group.setdefault(peer_backend, []).append((peer_version, candidate))
            group[backend] = [(version, target)]
            # The upstream cache is keyed only by system, not spec root.
            # Never reuse base GPU specifications for the head comparison.
            cross._SPEC_CACHE.clear()
            anomalies, diagnostics, _ = cross._check_table_group(
                ((system, filename), group),
                anomaly_factor=3.0,
                mono_tolerance=0.7,
                spike_factor=3.0,
                min_bucket_points=5,
                noise_floor=0.03,
                spec_root=root / SYSTEMS,
                fingerprint_factor=None,
            )
            by_side[side] = json.loads(json.dumps(anomalies, default=cross._jsonable, allow_nan=False))
            if side == "head":
                gaps.extend(
                    f"{path}: {item}"
                    for item in diagnostics
                    if item["kind"] in {"unsupported_schema", "schema_mismatch"}
                )
                findings.extend({"file": path, **item} for item in by_side[side])
            if filename == "gemm_perf.parquet" and cross._load_gpu_spec(root / SYSTEMS, system) is None:
                gaps.append(f"{side}/{path}: no usable GPU spec for speed-of-light checking")
        # This exact-base comparison is diagnostic ONLY. Never suppress head
        # findings just because they existed at base, or claim a reviewed baseline.
        baseline = cross.snapshot_baseline([item for item in by_side["base"] if item["kind"] in cross._GATE_THRESHOLDS])
        gated = [item for item in by_side["head"] if item["kind"] in cross._GATE_THRESHOLDS]
        _breaches, counts, known = cross.evaluate_gate(gated, dict.fromkeys(cross._GATE_THRESHOLDS, 0), baseline)
        comparisons.append(
            {
                "file": path,
                "base_findings": by_side["base"],
                "head_findings": by_side["head"],
                "diagnostic_new_counts": dict(counts),
                "diagnostic_known_count": known,
            }
        )
    gaps.extend(
        [
            "No reviewed committed anomaly baseline; exact-base diagnostic counts suppress nothing",
            "Extrapolation and non-GEMM physical bounds are not covered",
        ]
    )
    if not selected:
        gaps.append("No directly affected tables; indirect changes need expanded numerical coverage")
    hard = [item for item in findings if item["kind"] in cross._GATE_THRESHOLDS]
    return result(
        "FAIL" if hard else "INCOMPLETE",
        f"Checked {len(selected)} exact-version slices; {len(hard)} anomaly findings (zero suppression in shadow mode)",
        findings=findings,
        coverage_gaps=sorted(set(gaps)),
        comparisons=comparisons,
    )
