# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict each exported top recommendation and compare its AIPerf metrics with the search result."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

out = {}
for name in sys.argv[1:]:
    rec = json.loads(Path(f"rec/{name}/recommendation.json").read_text())
    cands = rec["candidates"]
    exported = yaml.safe_load(Path(f"rec/{name}/recommendations/0001.yaml").read_text())
    matches = [c for c in cands if c.get("prediction_config") == exported]
    assert len(matches) == 1, len(matches)
    top = matches[0]
    metrics = top.get("metrics") or top.get("candidate", {}).get("metrics", {})
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "-c",
            f"rec/{name}/recommendations/0001.yaml",
            "--format",
            "json",
            "--output-dir",
            f"rec/{name}/predict",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    pred = json.loads(proc.stdout[proc.stdout.index("{") :])
    keys = sorted(k for k in metrics if k.startswith("aiperf."))
    diff = max(abs(pred[k] - metrics[k]) / max(abs(metrics[k]), 1e-12) for k in keys)
    out[name] = {
        "counts": rec["counts"],
        "top_concurrency": exported["traffic"]["load"]["concurrency"],
        "candidates": [
            (
                c["traffic"]["load"]["concurrency"] if (c := x.get("prediction_config")) else None,
                x["status"],
                x.get("reason"),
            )
            for x in cands
        ],
        "evaluation": exported.get("evaluation"),
        "aiperf_keys": len(keys),
        "predict_vs_search_max_rel_diff": diff,
        "user_tps_p50": metrics.get("aiperf.output_token_throughput_per_user.p50"),
        "throughput_per_gpu": top.get("score"),
    }
print(json.dumps(out, indent=1))
