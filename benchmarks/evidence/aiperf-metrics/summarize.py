# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize compare.json files into Markdown and add real-data diagnostics."""

import glob
import json
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path

rows = []
for path in sorted(glob.glob("work/*/*/compare.json")):
    r = json.loads(Path(path).read_text())
    run_dir = f"runs/{r['model'].split('/')[1]}/{r['run']}"
    chunk, per_gap, per_req = Counter(), [], []
    for line in Path(run_dir, "profile_export_raw.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["metadata"].get("benchmark_phase") != "profiling":
            continue
        ts, first_tokens = [], None
        for resp in rec["responses"]:
            for pkt in resp["packets"]:
                try:
                    d = json.loads(pkt["value"])
                except ValueError:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    ts.append(resp["perf_ns"] / 1e6)
                    if first_tokens is None and d.get("usage"):
                        first_tokens = d["usage"]["completion_tokens"]
        chunk[first_tokens] += 1
        gaps = [b - a for a, b in pairwise(ts)]
        per_gap += [1000 / g for g in gaps if g > 0]
        if len(ts) > 1 and ts[-1] > ts[0]:
            per_req.append(1000 / ((ts[-1] - ts[0]) / (len(ts) - 1)))
    r["first_chunk_tokens"] = dict(chunk)
    r["legacy_per_gap_tps_mean"] = statistics.fmean(per_gap) if per_gap else None
    r["request_tps_mean_from_chunks"] = statistics.fmean(per_req) if per_req else None
    rows.append(r)

print(
    "| Model | Run | Records | Errors | Calc max rel diff | First-chunk tokens (count) "
    "| Per-gap TPS mean | AIPerf user TPS avg |"
)
print("|---|---|---:|---:|---:|---|---:|---:|")
for r in rows:
    print(
        f"| {r['model']} | {r['run']} | {r['records']} | {r['errors']} | {r['calc_max_rel_diff']:.1e} | "
        f"{r['first_chunk_tokens']} | {r['legacy_per_gap_tps_mean'] or float('nan'):.1f} | "
        f"{r['prediction']['output_token_throughput_per_user.avg'][1]:.1f} |"
    )
print()
print("| Model | Run | Metric | AIPerf avg / p50 / p99 | AISimulate avg / p50 / p99 | Rel. error avg / p50 / p99 |")
print("|---|---|---|---|---|---|")
for r in rows:
    for name, label in [
        ("time_to_first_token", "TTFT ms"),
        ("request_latency", "Latency ms"),
        ("inter_token_latency", "ITL ms"),
        ("output_token_throughput_per_user", "User TPS"),
    ]:
        v = [r["prediction"][f"{name}.{s}"] for s in ("avg", "p50", "p99")]
        print(
            f"| {r['model'].split('/')[1]} | {r['run']} | {label} | "
            + " / ".join(f"{x[1]:.2f}" for x in v)
            + " | "
            + " / ".join(f"{x[0]:.2f}" for x in v)
            + " | "
            + " / ".join(f"{x[2]:+.0%}" for x in v)
            + " |"
        )
Path("summary.json").write_text(json.dumps(rows, indent=1))
