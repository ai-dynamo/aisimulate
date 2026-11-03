#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare retained full-batch output-ready cadence with canonical op costs."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cadence(events, identities, target=128):
    """Keep every adjacent group in accounting; select only full-batch coverage."""
    by_request = defaultdict(list)
    groups = defaultdict(dict)
    for event in events:
        rid = event["rid"]
        if rid not in identities or event["rank"] != 0:
            continue
        at, count = int(event["at_ns"]), event["output_tokens"]
        if type(count) is not int or count < 0 or rid in groups[at]:
            raise ValueError("Invalid or duplicate output-ready counter")
        by_request[rid].append((at, count))
        groups[at][rid] = event
    if set(by_request) != identities:
        raise ValueError("Missing planned server-ready identity")
    excess = 0
    for values in by_request.values():
        if values != sorted(values) or any(b[1] < a[1] for a, b in pairwise(values)):
            raise ValueError("Counter or clock moved backward")
        if not any(count > 0 for _, count in values) or values[-1][1] < target:
            raise ValueError("Missing positive or terminal output counter")
        excess += max(0, values[-1][1] - target)
    start = max(next(at for at, count in rows if count > 0) for rows in by_request.values())
    end = min(next(at for at, count in rows if count >= target) for rows in by_request.values())
    excluded = Counter()
    retained = []
    ordered = sorted(groups.items())
    for (left, a), (right, b) in pairwise(ordered):
        if left < start or right >= end:
            excluded["outside_common_nonterminal_coverage"] += 1
        elif set(a) != identities or set(b) != identities:
            excluded["incomplete_membership"] += 1
        elif any(row["batch_requests"] != len(identities) for row in [*a.values(), *b.values()]):
            excluded["different_batch_size"] += 1
        elif any(b[rid]["output_tokens"] <= a[rid]["output_tokens"] for rid in identities):
            excluded["no_positive_progress"] += 1
        else:
            retained.append((right - left) / 1e6)
    result = {"requests": len(identities), "groups": len(groups), "adjacent_intervals": max(0, len(groups) - 1),
              "retained_intervals": len(retained), "excluded_intervals": dict(excluded),
              "generated_excess_tokens": excess, "common_coverage_ms": max(0, end - start) / 1e6}
    if len(retained) + sum(excluded.values()) != result["adjacent_intervals"]:
        raise ValueError("Incomplete interval accounting")
    if retained:
        result["cadence_ms"] = {"mean": statistics.mean(retained), "median": statistics.median(retained),
                                "min": min(retained), "max": max(retained),
                                "p95": sorted(retained)[math.ceil(0.95 * len(retained)) - 1]}
    else:
        result["unavailable"] = "No adjacent full-batch groups in common nonterminal coverage"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--canonical-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.controls.glob("scheduler-*.jsonl")) + [args.controls / "client/requests.jsonl",
                                                            args.canonical_receipt, Path(__file__)]
    manifest = {"schema": "aisimulate.glm-rsi.cadence-inputs.v1",
                "files": [{"path": str(path.resolve()), "sha256": checksum(path), "bytes": path.stat().st_size}
                          for path in paths],
                "cases": [{"repeat": repeat, "context": context, "lanes": lanes}
                          for repeat in (0, 1) for context in (8192, 131072) for lanes in (8, 14)]}
    if args.freeze:
        with args.manifest.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        return
    if json.loads(args.manifest.read_bytes()) != manifest:
        raise ValueError("Frozen cadence inputs changed")
    if args.output is None or args.output.exists():
        raise ValueError("Requires a new output file")
    observations = [json.loads(line) for line in (args.controls / "client/requests.jsonl").read_text().splitlines()]
    by_id = {row["rid"]: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError("Duplicate measured identity")
    events = [json.loads(line) for path in args.controls.glob("scheduler-*.jsonl")
              for line in path.read_text().splitlines()]
    config = ForwardPassPerfModelConfig(**json.loads(args.canonical_receipt.read_bytes())["canonical_config"])
    if (config.tp != 8 or config.attention_dp != 1 or config.backend_version != "0.5.21"
            or config.nextn != 3 or config.estimation_mode != "op_level" or config.database_mode != "SILICON"
            or config.fallback_policy != "deny"):
        raise ValueError("Not the frozen canonical model")
    model = RustForwardPassPerfModel.best_available(config)
    result = {"schema": "aisimulate.glm-rsi.cadence-audit.v1", "manifest_sha256": checksum(args.manifest),
              "scope": "Instrumented server-ready cadence, not pure GPU time, exact KV geometry or request MAPE. "
                       "Historical SGLang0.5.20 versus database0.5.21; no score or timing correction.", "cases": []}
    try:
        for case in manifest["cases"]:
            repeat, context, lanes = (case[key] for key in ("repeat", "context", "lanes"))
            identities = {f"rsi-r{repeat}-c{context}-o128-l{lanes}-i{lane}" for lane in range(lanes)}
            for rid in identities:
                row = by_id[rid]
                meta = row["events"][-1]["meta"]
                if (row["status"] != "completed" or row["input_tokens"] != context
                        or row["events"][-1]["output_tokens"] != 128 or meta.get("prompt_tokens") != context
                        or meta.get("cached_tokens") != 0):
                    raise ValueError("Measured input/output/cache/status qualification failed")
            item = {**case, **cadence(events, identities), "canonical_queries": []}
            for kv in (context, context + 132):
                metrics = [{"scheduled_requests": {"num_decode_requests": lanes * 4,
                            "sum_decode_kv_tokens": lanes * 4 * kv}}]
                detail = model.estimate_forward_pass_detailed(metrics)
                if not math.isfinite(detail["latency_ms"]) or detail["latency_ms"] <= 0:
                    raise ValueError("Invalid canonical latency")
                item["canonical_queries"].append({"incoming_kv_bound": kv, "metrics_by_rank": metrics,
                                                  "detail": detail})
            result["cases"].append(item)
    finally:
        model.close()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
