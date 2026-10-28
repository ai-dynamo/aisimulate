# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible standalone FPM cost probe; source and binary hashes accompany results.

Prepare with the candidate SDK, then measure each arm in a fresh process using
its own PYTHONPATH and native extension. Synthetic inputs exercise 17 aggregate
features, 150 trees, and 10,000 immutable anchors. These timings do not establish
accuracy on a serving workload or qualify engine replay integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import statistics
import time
from pathlib import Path

IDENTITY = {
    "model": "benchmark/private-model",
    "system": "benchmark_cpu",
    "backend": "vllm",
    "worker_type": "decode",
    "estimation_mode": "fpm_regression",
}
ONLINE = {
    "adaptation": "online",
    "recent_window": 64,
    "update_every": 1,
    "history_weight": 0.1,
    "history_weight_ramp_observations": 32,
    "history_bin_grouping": True,
    "derivative_refresh": "lazy",
    "eager_refresh_interval": 32,
    "split_sampling_rate": 0.5,
    "split_rank_tolerance": 0.01,
}
CORRECTION = {
    "enabled": True,
    "mode": "hybrid",
    "sampling": {"max_observations": 32, "bins_per_axis": [2, 2]},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(count, seed):
    rng = random.Random(seed)
    for index in range(count):
        n = rng.randint(1, 256)
        mean = rng.randint(128, 16000)
        y = 0.6 + 0.015 * n + 0.0000004 * n * mean
        yield [
            {
                "version": 1,
                "worker_id": "benchmark-worker",
                "counter_id": index,
                "wall_time": y / 1000,
                "scheduled_requests": {
                    "num_decode_requests": n,
                    "sum_decode_kv_tokens": n * mean,
                    "var_decode_kv_tokens": 0.0,
                },
            }
        ]


def prepare(root, history_rows):
    from aisimulate_core.sdk import fpm_learned

    root.mkdir(parents=True, exist_ok=True)
    train = list(rows(history_rows, 0))
    seed = root / "seed"
    fpm_learned.train(
        train,
        "decode",
        features=fpm_learned.FPM5_FEATURE_NAMES,
        max_iter=150,
        learning_rate=0.2,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        early_stopping=False,
        online_checkpoint_dir=seed,
        identity=IDENTITY,
    )
    (root / "queries.json").write_text(json.dumps(list(rows(256, 1)), allow_nan=False))
    print(
        json.dumps(
            {
                "seed_manifest_sha256": digest(seed / "manifest.json"),
                "queries_sha256": digest(root / "queries.json"),
                "training_rows": len(train),
            }
        )
    )


def measure(root, arm):
    import aisimulate._runtime as native
    from aisimulate_core.sdk import RustForwardPassPerfModel

    queries = json.loads((root / "queries.json").read_text())
    artifact = root / "seed/model.json"
    config = {
        **IDENTITY,
        "estimation_mode": "fpm_learned",
        "estimator_config": {
            "correction": CORRECTION if arm != "online" else {"enabled": False},
            "fpm_learned": {"artifact_path": str(artifact), "adaptation": "frozen"},
        },
    }
    if arm == "online":
        learned = {**ONLINE, "checkpoint_path": str(root / "seed")}
        config["estimator_config"]["fpm_learned"] = learned
    start = time.perf_counter_ns()
    if arm == "base":
        model = RustForwardPassPerfModel.from_learned(
            artifact,
            {"correction_mode": "hybrid", "max_observations": 32, "bucket_count": 4},
        )
    else:
        model = RustForwardPassPerfModel.best_available(config)
    setup_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    first = model.estimate_forward_pass_time_ms(queries[0])
    cold_us = (time.perf_counter_ns() - start) / 1000
    # One full pass warms every query. The hash proves identical initial work.
    initial = [model.estimate_forward_pass_time_ms(q) for q in queries]
    assert all(x is not None and math.isfinite(x) and x > 0 for x in initial)
    warm = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(4):
            for q in queries:
                model.estimate_forward_pass_time_ms(q)
        warm.append((time.perf_counter_ns() - start) / (4 * len(queries)) / 1000)
    updates = []
    for q, value in zip(queries, initial, strict=True):
        q[0]["wall_time"] = 1.3 * value / 1000
        start = time.perf_counter_ns()
        model.tune_with_fpms([q])
        updates.append((time.perf_counter_ns() - start) / 1000)
    diagnostics = None if arm == "base" else model.learned_online_diagnostics()
    if arm == "online":
        state = diagnostics["stores"]["pure_decode"]
        assert state["accepted_observations"] == 256
        assert state["recent_rows"] == 64 and state["pending_rows"] == 0
        assert state["publications"] == 256
    final = [model.estimate_forward_pass_time_ms(q) for q in queries]
    answer = {
        "arm": arm,
        "binary_sha256": digest(native.__file__),
        "seed_manifest_sha256": digest(root / "seed/manifest.json"),
        "queries_sha256": digest(root / "queries.json"),
        "setup_ms": setup_ms,
        "cold_query_us": cold_us,
        "first_prediction_ms": first,
        "initial_predictions_ms": initial,
        "final_predictions_ms": final,
        "initial_prediction_sha256": hashlib.sha256(
            json.dumps(initial).encode()
        ).hexdigest(),
        "final_prediction_sha256": hashlib.sha256(
            json.dumps(final).encode()
        ).hexdigest(),
        "warm_query_us_rounds": warm,
        "warm_query_us_median": statistics.median(warm),
        "update_us_median": statistics.median(updates),
        "update_us_p95": sorted(updates)[int(0.95 * (len(updates) - 1))],
        "update_us_max": max(updates),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "diagnostics": diagnostics,
        "config": config if arm != "base" else "PR493 compatibility adapter",
    }
    print(json.dumps(answer, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "base", "frozen", "online"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--history-rows", type=int, default=10000)
    args = parser.parse_args()
    prepare(args.root, args.history_rows) if args.action == "prepare" else measure(
        args.root, args.action
    )
