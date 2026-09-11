# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce admission-predicate counts from six immutable native input snapshots.

Only input identities, shapes and historical status labels are inspected. No
latency predictor, timing/error reduction or new measurement admission is run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

INPUT_SNAPSHOT_COMMIT = "6a3e9d7f965b87fcf3d9ff4a9d207679db4d49af"
PREDICATE_FIX_COMMIT = "3e2e194ca1c1f51a441a690821cd56c3f4e09bda"
PINNED_COMPRESSED_SHA256 = {
    "off/core": "a82402ae0a2f181171db55e44cba162f9de462c7b98790c9a76544572e88fbfe",
    "off/field": "0ffe3962927490e9bc45191ab481f6eb53be67cd6ec9309161340b637d98d45f",
    "off/service": "b49e624ed5bf68cac22695c6267ef91aaaa1f4797173af16b7f3b4291d155502",
    "on/core": "9c23e7cce528d04f5bf1aa75f026498edc03bb430a64792edc8dba5cf44e56be",
    "on/field": "da728c5e66f251f6cc03721d39702d543418635feeec3bbe5f2cb7cc08717506",
    "on/service": "a60bfeb41fe729dedb818f95f5db2d3901a7958f00180f352e912b4e2a149e5c",
}
EXPECTED_COUNTS = {
    "off/core": 15756,
    "off/field": 3503,
    "off/service": 3509,
    "on/core": 39393,
    "on/field": 3503,
    "on/service": 3504,
}
COUNT_KEYS = (
    "native_inputs",
    "old_predicate_rejected",
    "new_predicate_rejected",
    "newly_rejected",
    "unchanged_admitted",
    "historically_predicted",
    "newly_rejected_historically_predicted",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def predicates(metrics, decoder_on):
    batch, tokens, variance = (metrics[k] for k in ("num_prefill_requests", "sum_prefill_tokens", "var_prefill_length"))
    require(type(batch) is int and batch >= 0, "invalid prefill request count")
    require(type(tokens) is int and tokens >= 0, "invalid fresh-token count")
    require(type(variance) in (int, float), "invalid variance type")
    invalid = not math.isfinite(variance) or variance < 0
    active = decoder_on and tokens > 0
    return active and (invalid or variance > 0), active and (invalid or batch > 1)


def audit(repo):
    sources, populations, seen = {}, {}, set()
    for scope, expected in EXPECTED_COUNTS.items():
        profile, population = scope.split("/")
        basename = "segments.json.gz" if scope == "on/core" else "trace.json.gz"
        relative = (
            f"data/experimental/deepseek-v41/gb300-fpm/{profile}-union-v3-tracewait2/reports/{population}/{basename}"
        )
        compressed = (repo / relative).read_bytes()
        require(digest(compressed) == PINNED_COMPRESSED_SHA256[scope], f"frozen input changed: {relative}")
        raw = gzip.decompress(compressed)
        document = json.loads(raw)
        sources[relative] = {
            "compressed_sha256": digest(compressed),
            "uncompressed_sha256": digest(raw),
            "bytes": len(compressed),
        }
        if scope == "on/core":
            cohorts = [(s["physical_run_id"], c) for s in document["segments"] for c in s["trace_cases"]]
        else:
            cohorts = [(document["run_id"], c) for c in document["cohorts"]]
        counts = Counter(dict.fromkeys(COUNT_KEYS, 0))
        geometries = Counter()
        for physical_run, cohort in cohorts:
            require(cohort.get("physical_run_id", physical_run) == physical_run, "physical lifecycle mismatch")
            for interval in cohort["intervals"]:
                native = interval["native_scheduled_requests"]
                key = (physical_run, interval["counter_id"])
                require(key not in seen, f"duplicate native counter: {key}")
                seen.add(key)
                old, new = predicates(native, profile == "on")
                require(
                    (old, new) == predicates(interval["scheduled_requests"], profile == "on"),
                    "normalization bridge changes admission predicate",
                )
                require(not (old and not new), "unexpected admission broadening in frozen inputs")
                predicted = interval["status"] == "predicted"
                counts.update(
                    dict(
                        zip(
                            COUNT_KEYS,
                            (
                                1,
                                old,
                                new,
                                new and not old,
                                not new,
                                predicted,
                                new and not old and predicted,
                            ),
                            strict=True,
                        )
                    )
                )
                if new and not old:
                    geometries[
                        (
                            native["num_prefill_requests"],
                            native["sum_prefill_tokens"],
                            native["sum_prefill_kv_tokens"],
                            native["var_prefill_length"],
                        )
                    ] += 1
        require(counts["native_inputs"] == expected, f"native input count mismatch: {scope}")
        populations[scope] = {
            **counts,
            "newly_rejected_geometries": [
                {"batch": k[0], "new_tokens": k[1], "prefix_tokens": k[2], "variance": k[3], "count": v}
                for k, v in sorted(geometries.items())
            ],
        }
    totals = {k: sum(p[k] for p in populations.values()) for k in COUNT_KEYS}
    require(totals["native_inputs"] == len(seen) == 69168, "native input inventory mismatch")
    return {
        "schema": "dsv41.fpm.frozen-input-predicate-audit.v1",
        "input_snapshot_commit": INPUT_SNAPSHOT_COMMIT,
        "predicate_reference_commits": {"old": INPUT_SNAPSHOT_COMMIT, "new": PREDICATE_FIX_COMMIT},
        "script_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Six frozen union native trace input populations; no prediction or error recalculation",
        "old_predicate": "Decoder ON and fresh prefill and (invalid variance or variance > 0)",
        "new_predicate": "Decoder ON and fresh prefill and (invalid variance or num_prefill_requests > 1)",
        "deduplication_key": ["physical_run_id", "counter_id"],
        "duplicates": 0,
        "native_and_prediction_predicates_equal": True,
        "sources": sources,
        "populations": populations,
        "totals": totals,
        "measurement_or_accuracy_update": False,
        "limits": [
            "Counts describe the declared admission predicates on preserved inputs only",
            "Newly rejected physical executions are not claimed to have heterogeneous extends",
            "No new predictor run, observation, accuracy or confidence interval",
            "Historical supported counts and accuracy denominators remain unchanged",
            "Raw worker logs and private measurements are not re-admitted by this audit",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.dumps(audit(args.repo), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(result)
    print(json.dumps(json.loads(result)["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
