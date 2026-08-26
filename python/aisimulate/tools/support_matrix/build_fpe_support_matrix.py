#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roll strict-native op-level FPE probe artifacts into the web support-matrix schema."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_APPLICATION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_APPLICATION_ROOT))

from tools.support_matrix.fpe_support_matrix import (
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    STATUS_PASS,
)

SYSTEM_ORDER = (
    "b200_sxm",
    "gb200",
    "b300_sxm",
    "gb300",
    "rtx_pro_6000_server",
    "h200_sxm",
    "h100_sxm",
    "l40s",
    "a100_sxm",
    "b60",
)
WEB_FIELDNAMES = (
    "HuggingFaceID",
    "Architecture",
    "System",
    "Backend",
    "Version",
    "Status",
    "ErrMsg",
    "Command",
    "Source",
    "FPEProbeCount",
    "FPETopologyCount",
    "FPEStatusCounts",
    "FPEPhaseLatencyMs",
    "SourceSHA",
)
TOPOLOGY_FIELDS = (
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
    "cp_size",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "nextn",
)


def _artifact_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("fpe_support_matrix.json"))
        elif path.name == "fpe_support_matrix.json":
            paths.add(path)
        else:
            raise ValueError(f"input must be a probe JSON or directory, got {path}")
    if not paths:
        raise ValueError("no fpe_support_matrix.json artifacts found")
    return sorted(paths)


def load_artifacts(inputs: Sequence[str | Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load compatible op-level FPE shard artifacts and reject mixed source identities."""
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for path in _artifact_paths(inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = payload["metadata"]
        identity = {key: current.get(key) for key in ("schema_version", "source_version", "source_sha", "workload")}
        if metadata is None:
            metadata = identity
        elif identity != metadata:
            raise ValueError(f"artifact metadata does not match the first shard: {path}")
        for row in payload["results"]:
            if row.get("forward_model") != "op_level":
                raise ValueError(f"non-op-level result in {path}: {row.get('forward_model')!r}")
            rows.append(row)
    assert metadata is not None
    return rows, metadata


def _topology(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in TOPOLOGY_FIELDS)


def _by_topology(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_topology(row)].append(row)
    return dict(grouped)


def _passing_topology(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]] | None:
    required_phases = {"prefill", "decode_start", "decode_end", "mixed"}
    for _identity, candidates in sorted(_by_topology(rows).items()):
        passed = {row["phase"] for row in candidates if row["status"] == STATUS_PASS}
        if required_phases <= passed:
            return candidates
    return None


def _complete_topology_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep topology candidates that were exercised across the complete FPE phase set."""
    grouped = _by_topology(rows)
    complete = {
        identity
        for identity, candidates in grouped.items()
        if any(str(row.get("phase")) == "mixed" for row in candidates)
    }
    return [row for row in rows if _topology(row) in complete] or list(rows)


def _failure_status(rows: Sequence[dict[str, Any]]) -> str:
    statuses = {row["status"] for row in rows}
    if statuses == {STATUS_HW_INCOMPATIBLE}:
        return STATUS_HW_INCOMPATIBLE
    if statuses == {STATUS_FRAMEWORK_INCOMPATIBLE}:
        return STATUS_FRAMEWORK_INCOMPATIBLE
    return "FAIL"


def _status_counts(rows: Sequence[dict[str, Any]]) -> str:
    counts = Counter(str(row["status"]) for row in rows)
    return ", ".join(f"{status}={counts[status]}" for status in sorted(counts))


def _latency_summary(rows: Sequence[dict[str, Any]]) -> str:
    values: dict[str, float] = {}
    for row in rows:
        latency = row.get("latency_ms")
        if row.get("status") == STATUS_PASS and latency is not None:
            values.setdefault(str(row["phase"]), float(latency))
    return ", ".join(f"{phase}={values[phase]:.6f}" for phase in sorted(values))


def _failure_summary(rows: Sequence[dict[str, Any]], *, metadata: dict[str, Any]) -> str:
    messages = sorted(
        {
            re.sub(
                r"(?:/[A-Za-z0-9_.-]+)+/python/aisimulate/",
                "<repo>/python/aisimulate/",
                " ".join(str(row.get("error_message", "")).split()),
            )
            for row in rows
            if str(row.get("error_message", "")).strip()
        }
    )
    summary = (
        f"Strict-native op-level FPE probes: {_status_counts(rows)}; "
        f"raw probes={len(rows)}; topologies={len(_by_topology(rows))}; "
        f"source_sha={metadata['source_sha']}"
    )
    if messages:
        summary += f". Representative failure: {messages[0][:1200]}"
    return summary


def _command(key: tuple[str, str, str, str, str]) -> str:
    model, _architecture, system, backend, version = key
    values = {
        "model": model,
        "system": system,
        "backend": backend,
        "backend-version": version,
    }
    args = ["python", "python/aisimulate/tools/support_matrix/generate_fpe_support_matrix.py"]
    for name, value in values.items():
        args.extend((f"--{name}", json.dumps(value)))
    args.extend(("--forward-model", "op_level", "--max-workers", "8"))
    return " ".join(args)


def build_web_rows(raw_rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Roll topology/phase rows into one mode-neutral FPE capability cell."""
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["model"]),
            str(row["architecture"]),
            str(row["system"]),
            str(row["backend"]),
            str(row["backend_version"]),
        )
        grouped[key].append(row)

    web_rows: list[dict[str, str]] = []
    for key, rows in sorted(grouped.items()):
        model, architecture, system, backend, version = key
        relevant = _complete_topology_rows(rows)
        selected = _passing_topology(relevant) or []
        passes = bool(selected)
        status = STATUS_PASS if passes else _failure_status(relevant)
        source = ",".join(sorted({str(row.get("source", "")) for row in selected if row.get("source")}))
        latency = _latency_summary(selected)
        web_rows.append(
            {
                "HuggingFaceID": model,
                "Architecture": architecture,
                "System": system,
                "Backend": backend,
                "Version": version,
                "Status": status,
                "ErrMsg": "" if passes else _failure_summary(relevant, metadata=metadata),
                "Command": _command(key),
                "Source": source,
                "FPEProbeCount": str(len(relevant)),
                "FPETopologyCount": str(len(_by_topology(relevant))),
                "FPEStatusCounts": _status_counts(relevant),
                "FPEPhaseLatencyMs": latency,
                "SourceSHA": str(metadata["source_sha"]),
            }
        )
    return web_rows


def write_web_matrix(rows: Sequence[dict[str, str]], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["System"]].append(row)

    unknown = sorted(set(grouped) - set(SYSTEM_ORDER))
    systems = [system for system in SYSTEM_ORDER if system in grouped] + unknown
    files: list[str] = []
    for system in systems:
        filename = f"{system}.csv"
        files.append(filename)
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=WEB_FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(grouped[system])
    index = {"files": files}
    (destination / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build split FPE CSVs for the existing support-matrix page")
    parser.add_argument("inputs", nargs="+", help="Probe JSON files or directories containing shard artifacts")
    parser.add_argument("--output-dir", required=True, help="Destination for index.json and per-system CSVs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_rows, metadata = load_artifacts(args.inputs)
    web_rows = build_web_rows(raw_rows, metadata)
    index = write_web_matrix(web_rows, args.output_dir)
    counts = Counter(row["Status"] for row in web_rows)
    print(
        f"raw_results={len(raw_rows)} web_rows={len(web_rows)} systems={len(index['files'])} "
        + " ".join(f"{status}={counts[status]}" for status in sorted(counts))
    )


if __name__ == "__main__":
    main()
