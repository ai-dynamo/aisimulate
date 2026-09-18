#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce PR #244's sanitized collection and parquet evidence without a GPU."""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "python/aisimulate/src/aisimulate_core/systems/data"
EVIDENCE = ROOT / "docs/data/pr244"
SOURCE = "1f29ee459882796db312b8a288c040b83edbb211"
SYSTEMS = ("b300_sxm", "gb200", "gb300", "h100_sxm", "h200_sxm")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plan_hash(ids):
    return sha256("\n".join(ids).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_cases():
    """Allowlist final checkpoint IDs/statuses; never export diagnostic text."""
    ledger = {"schema_version": 1, "source_revision": SOURCE, "plans": {}, "systems": {}}
    for system in SYSTEMS:
        prefix = f"{DATA.relative_to(ROOT)}/{system}/vllm-0.25.0-"
        raw = {
            name: subprocess.check_output(["git", "show", f"{SOURCE}:{prefix}{name}"], cwd=ROOT)
            for name in ("collection-report.json", "failures.json.gz")
        }
        report = json.loads(raw["collection-report.json"])
        failures = json.loads(gzip.decompress(raw["failures.json.gz"]))
        attempts = []
        for attempt in report["attempts"]:
            group = failures[attempt["id"]]
            done, failed = set(), set()
            for checkpoint in group["checkpoints"].values():
                done.update(checkpoint["done"])
                failed.update(checkpoint["failed"])
            require(not done & failed, f"Conflicting final outcomes: {system}/{attempt['id']}")
            # B300's archived plan was omitted. Its final checkpoint union must
            # match BOTH the independently recorded retained count and ID digest.
            ids = sorted(group.get("case_plan", {}).get("task_ids", done | failed))
            digest = plan_hash(ids)
            require(digest == attempt["task_ids_sha256"], "Original retained-plan digest mismatch")
            require(len(ids) == attempt["expected"] == len(set(ids)), "Original retained-plan count mismatch")
            require(done | failed <= set(ids), "Outcome outside the retained plan")
            for task in ids:
                require(
                    re.fullmatch(r"(?:vllm\.[a-z0-9_]+:[a-z0-9_]+:\[[a-zA-Z0-9_/., '\-\[\]]*\]|mhc_[a-z0-9_]+)", task),
                    f"Unexpected task-ID format: {system}/{attempt['id']}",
                )
            statuses = "".join("D" if task in done else "F" if task in failed else "U" for task in ids)
            counts = {
                "expected": len(ids),
                "done": statuses.count("D"),
                "failed": statuses.count("F"),
                "unattempted": statuses.count("U"),
            }
            require(all(counts[key] == attempt[key] for key in counts), "Original outcome counts mismatch")
            ledger["plans"][digest] = ids
            attempts.append(
                {
                    "id": attempt["id"],
                    "op": attempt["op"],
                    "plan_sha256": digest,
                    "plan_source": "archived_case_plan" if "case_plan" in group else "verified_checkpoint_union",
                    "reported_counts": counts,
                    "outcomes": statuses,
                }
            )
        ledger["systems"][system] = {
            "source_artifact_sha256": {name: sha256(content) for name, content in raw.items()},
            "attempts": attempts,
            "reported_tables": {
                name: {key: table[key] for key in ("rows", "sha256")} for name, table in report["tables"].items()
            },
        }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    (EVIDENCE / "case-outcomes.json.gz").write_bytes(gzip.compress(payload, mtime=0))


def summarize_cases(ledger):
    for digest, ids in ledger["plans"].items():
        require(ids == sorted(set(ids)) and plan_hash(ids) == digest, "Retained-plan digest/order mismatch")
    systems = {}
    for system in SYSTEMS:
        attempts = []
        seen = set()
        for attempt in ledger["systems"][system]["attempts"]:
            require(attempt["id"] not in seen, "Duplicate attempt ID")
            seen.add(attempt["id"])
            ids = ledger["plans"][attempt["plan_sha256"]]
            outcomes = attempt["outcomes"]
            require(len(outcomes) == len(ids) and set(outcomes) <= set("DFU"), "Invalid outcome vector")
            counts = {
                "expected": len(ids),
                "done": outcomes.count("D"),
                "failed": outcomes.count("F"),
                "unattempted": outcomes.count("U"),
            }
            require(counts == attempt["reported_counts"], "Reported outcome counts mismatch")
            attempts.append({key: attempt[key] for key in ("id", "op", "plan_sha256", "plan_source")} | counts)
        systems[system] = {
            "cases": {key: sum(a[key] for a in attempts) for key in ("expected", "done", "failed", "unattempted")},
            "attempts": attempts,
        }
    return systems


def summarize_table(path):
    table = pq.read_table(path)
    # Physical row identity used by the original collection validation; this is
    # not a claim about the engine's normalized lookup keys.
    keys = [name for name in table.column_names if name not in ("device", "latency")]
    columns = {}
    for field in table.schema:
        column = table[field.name]
        summary = {"type": str(field.type), "nulls": column.null_count, "distinct": pc.count_distinct(column).as_py()}
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            summary.update(pc.min_max(column).as_py())
        else:
            summary["values"] = sorted(value for value in pc.unique(column).to_pylist() if value is not None)
        columns[field.name] = summary
    latency = table["latency"]
    key_rows = zip(*(table[key].to_pylist() for key in keys), strict=True)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
        "rows": table.num_rows,
        "sidecar_sha256": sha256(path.with_name("collection_meta.yaml").read_bytes()),
        "physical_key_columns": keys,
        "latency_unit": "ms",
        "columns": columns,
        "anomalies": {
            "duplicate_physical_keys": table.num_rows - len(set(key_rows)),
            "null_cells": sum(column.null_count for column in table.columns),
            "nonfinite_latency": pc.sum(pc.invert(pc.is_finite(latency))).as_py(),
            "nonpositive_latency": pc.sum(pc.less_equal(latency, 0)).as_py(),
        },
    }


def build_manifest():
    raw = (EVIDENCE / "case-outcomes.json.gz").read_bytes()
    ledger = json.loads(gzip.decompress(raw))
    require(ledger["source_revision"] == SOURCE, "Unexpected source revision")
    systems = summarize_cases(ledger)
    for system, summary in systems.items():
        tables = {path.name: summarize_table(path) for path in sorted((DATA / system).glob("*/vllm/0.25.0/*.parquet"))}
        reported = ledger["systems"][system]["reported_tables"]
        require(set(tables) == set(reported), f"Table inventory drift: {system}")
        for name, table in tables.items():
            require(
                all(count == 0 for count in table["anomalies"].values()),
                f"Published table contains anomalies: {system}/{name}: {table['anomalies']}",
            )
            require(
                all(table[key] == reported[name][key] for key in ("rows", "sha256")),
                f"Published table differs from collection report: {system}/{name}",
            )
        summary.update({"table_count": len(tables), "rows": sum(t["rows"] for t in tables.values()), "tables": tables})
    return {
        "schema_version": 1,
        "source_revision": SOURCE,
        "collector_revision": "cbaf51b64fa460e5ec6146bde407a4c64958212d",
        "vllm_runtime_revision": "dd10e03f95f94edbea1975c67ace3a35ec9a8a40",
        "case_outcomes_sha256": sha256(raw),
        "systems": systems,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract-cases", action="store_true", help="Recover sanitized evidence from pinned Git history"
    )
    parser.add_argument("--write", action="store_true", help="Write the manifest instead of checking it")
    args = parser.parse_args()
    if args.extract_cases:
        extract_cases()
    manifest = build_manifest()
    path = EVIDENCE / "manifest.json"
    if args.write:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        require(manifest == json.loads(path.read_text()), "Manifest drift; inspect changes before using --write")
    for system, summary in manifest["systems"].items():
        print(f"{system}: {summary['table_count']} tables, {summary['rows']} rows, {summary['cases']}")


if __name__ == "__main__":
    main()
