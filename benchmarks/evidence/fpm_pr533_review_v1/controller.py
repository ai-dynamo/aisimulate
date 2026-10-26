#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired frozen-anchor-cache benchmark using the maintained prefix probe."""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, rounds=5, updates=32):
    binaries = {arm: root / f"{arm}-probe" for arm in ("baseline", "candidate")}
    hashes = {arm: digest(path) for arm, path in binaries.items()}
    assert hashes["baseline"] != hashes["candidate"]
    cases = [("small_tail", "small-fixture", "1"), ("large_tail", "fixture", "1"), ("large_all", "fixture", "all")]
    equality_keys = [
        "config",
        "seed_manifest_sha256",
        "queries_sha256",
        "initial_predictions_sha256",
        "pre_update_trajectory_sha256",
        "final_predictions_sha256",
        "final_artifact_sha256",
        "semantic_diagnostics_sha256",
        "prediction_count",
        "updates",
    ]
    references = {}
    samples = []
    results = root / "results"
    results.mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for round_index in range(rounds):
        for name, fixture, tail in cases if round_index % 2 == 0 else cases[::-1]:
            for phase in ("timing", "allocation"):
                for arm in ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline"):
                    command = [
                        str(binaries[arm]),
                        str(root / fixture / "seed"),
                        str(root / fixture / "queries.json"),
                        tail,
                        str(updates),
                        phase,
                    ]
                    raw = subprocess.check_output(command, env=env)
                    path = results / f"{round_index}-{name}-{phase}-{arm}.json"
                    path.write_bytes(raw)
                    value = json.loads(raw)
                    assert value["binary_sha256"] == hashes[arm]
                    comparable = {key: value[key] for key in equality_keys}
                    assert comparable == references.setdefault(name, comparable)
                    store = value["final_diagnostics"]["stores"]["pure_decode"]
                    assert store["accepted_observations"] == store["publications"] == store["eager_updates"] == updates
                    assert store["pending_rows"] == store["update_failures"] == 0
                    sample = {
                        "round": round_index,
                        "case": name,
                        "phase": phase,
                        "arm": arm,
                        "result_sha256": digest(path),
                        "setup_ms": value["setup_ns"] / 1e6,
                        "retained_bytes": value["final_diagnostics"]["training_state_bytes"],
                    }
                    if phase == "timing":
                        sample["p50_ms"] = statistics.median(value["durations_ns"]) / 1e6
                        sample["p95_ms"] = sorted(value["durations_ns"])[int(0.95 * (updates - 1))] / 1e6
                    else:
                        sample["requested_bytes_median"] = statistics.median(
                            a["requested_bytes"] for a in value["allocations"]
                        )
                        sample["extra_live_bytes_max"] = max(
                            a["peak_additional_live_requested_bytes"] for a in value["allocations"]
                        )
                    samples.append(sample)
            print(f"completed round {round_index + 1}/{rounds}: {name}", flush=True)
    summary = {}
    for name, _, _ in cases:
        selected = [s for s in samples if s["case"] == name]
        summary[name] = {}
        for arm in binaries:
            rows = [s for s in selected if s["arm"] == arm]
            summary[name][arm] = {
                key: statistics.median(s[key] for s in rows if key in s)
                for key in (
                    "setup_ms",
                    "p50_ms",
                    "p95_ms",
                    "retained_bytes",
                    "requested_bytes_median",
                    "extra_live_bytes_max",
                )
            }
        changes = []
        for round_index in range(rounds):
            pair = {s["arm"]: s for s in selected if s["round"] == round_index and s["phase"] == "timing"}
            changes.append(100 * (pair["candidate"]["p50_ms"] / pair["baseline"]["p50_ms"] - 1))
        summary[name]["paired_p50_percent"] = {
            "median": statistics.median(changes),
            "min": min(changes),
            "max": max(changes),
        }
    assert hashes == {arm: digest(path) for arm, path in binaries.items()}
    report = {
        "schema_version": 1,
        "host": platform.platform(),
        "rounds": rounds,
        "updates_per_process": updates,
        "binary_sha256": hashes,
        "exact_equality": references,
        "summary": summary,
        "samples": samples,
    }
    (root / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
