#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate complete, same-artifact FPE reports before nightly staging."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PHASES = {"prefill", "decode_start", "decode_end", "mixed"}
IDENTITY = ("model", "system", "backend", "backend_version", "forward_model")
TOPOLOGY = (
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
    "attention_backend",
)
STATUSES = {
    "PASS",
    "SDK_UNREPRESENTABLE",
    "PERF_DATA_MISSING",
    "MODEL_UNSUPPORTED",
    "HW_INCOMPATIBLE",
    "FRAMEWORK_INCOMPATIBLE",
    "BUILD_FAILED",
    "QUERY_FAILED",
}


def qualify(
    root: Path,
    *,
    expected_shards: list[dict[str, str]],
    expected_sha: str,
    expected_wheel_sha256: str,
    required_probes: list[dict[str, str]],
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("expected source SHA must be a full commit")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_wheel_sha256):
        raise ValueError("expected wheel SHA256 must be a full digest")
    expected = {(s["system"], s["backend"]) for s in expected_shards}
    if not expected or len(expected) != len(expected_shards):
        raise ValueError("expected shards must be nonempty and unique")
    if not required_probes:
        raise ValueError("required probes must not be empty")
    required = {tuple(p[k] for k in IDENTITY) for p in required_probes}
    if len(required) != len(required_probes):
        raise ValueError("required probes must be unique")
    found: set[tuple[str, str]] = set()
    qualified: set[tuple] = set()
    counts: Counter = Counter()
    shared_metadata = None
    for path in sorted(root.rglob("fpe_support_matrix.json")):
        payload = json.loads(path.read_text())
        metadata, rows = payload["metadata"], payload["results"]
        if metadata.get("schema_version") != 1 or metadata.get("source_sha") != expected_sha:
            raise ValueError(f"wrong source/schema identity: {path}")
        if metadata.get("wheel_sha256") != expected_wheel_sha256:
            raise ValueError(f"wrong wheel identity: {path}")
        identity = {k: metadata[k] for k in ("source_version", "workload")}
        if shared_metadata is not None and identity != shared_metadata:
            raise ValueError(f"inconsistent version or workload: {path}")
        shared_metadata = identity
        if not rows:
            raise ValueError(f"empty shard: {path}")
        shards = {(r["system"], r["backend"]) for r in rows}
        if len(shards) != 1 or shards & found or not shards <= expected:
            raise ValueError(f"duplicate, mixed, or unexpected shard: {path}")
        found.update(shards)
        phases: dict[tuple, set[str]] = defaultdict(set)
        passed: dict[tuple, set[str]] = defaultdict(set)
        for row in rows:
            key = tuple(row[k] for k in (*IDENTITY, *TOPOLOGY, "roles"))
            phase, status = row["phase"], row["status"]
            if row["source_sha"] != expected_sha or row["source_version"] != metadata["source_version"]:
                raise ValueError(f"wrong row source identity: {path}")
            if row["forward_model"] != "op_level" or status not in STATUSES or phase not in PHASES:
                raise ValueError(f"invalid probe result: {path}")
            if phase in phases[key]:
                raise ValueError(f"duplicate probe phase: {path}")
            phases[key].add(phase)
            counts[status] += 1
            if status in {"BUILD_FAILED", "QUERY_FAILED"}:
                raise ValueError(f"unexpected native probe failure: {key}: {status}")
            if status == "PASS":
                latency = row["latency_ms"]
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or latency <= 0
                ):
                    raise ValueError(f"invalid passing latency: {path}")
                passed[key].add(phase)
        for key, actual_phases in phases.items():
            roles = set(key[-1].split("|"))
            if not roles or not roles <= {"agg", "prefill", "decode"}:
                raise ValueError(f"invalid probe roles: {path}")
            wanted = (
                PHASES
                if "agg" in roles
                else ({"prefill"} if "prefill" in roles else set())
                | ({"decode_start", "decode_end"} if "decode" in roles else set())
            )
            if actual_phases != wanted:
                raise ValueError(f"incomplete topology phases: {path}")
        if metadata["plan_count"] != len(phases):
            raise ValueError(f"plan count does not match probe results: {path}")
        qualified.update(k[: len(IDENTITY)] for k, p in passed.items() if p == PHASES)
    if found != expected:
        raise ValueError(f"missing shards: {sorted(expected - found)}")
    if missing := required - qualified:
        raise ValueError(f"required known-good probes no longer pass: {sorted(missing)}")
    return {
        "schema_version": 1,
        "source_sha": expected_sha,
        "wheel_sha256": expected_wheel_sha256,
        "shard_count": len(found),
        "required_probe_count": len(required),
        "status_counts": dict(counts),
        "qualification": "complete_native_fpe_reports_and_required_probes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-shards", required=True, help="JSON array from the discovery job")
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--required-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(
        args.root,
        expected_shards=json.loads(args.expected_shards),
        expected_sha=args.expected_sha,
        expected_wheel_sha256=args.expected_wheel_sha256,
        required_probes=json.loads(args.required_probes.read_text())["probes"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
