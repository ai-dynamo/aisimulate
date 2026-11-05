# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare AISimulate AIPerf metrics with a pinned AIPerf run.

Usage: compare.py RUN_DIR MODEL REVISION CONCURRENCY PER_CHUNK(0|1) WORK_DIR
1. Calculation agreement: rebuild records from profile_export_raw.jsonl and
   compare aisimulate.aiperf_metrics.summarize_requests with AIPerf's aggregate.
2. Prediction error: replay the observed request lengths, in dispatch order,
   with `aisimulate predict` and compare the same statistics.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

run_dir, model, rev, conc, per_chunk, work = sys.argv[1:7]
conc, per_chunk = int(conc), per_chunk == "1"
from aisimulate.aiperf_metrics import summarize_requests

NAMES = ("time_to_first_token", "request_latency", "inter_token_latency", "output_token_throughput_per_user")
STATS = ("avg", "p50", "p90", "p95", "p99")


def content_chunks(rec):
    out = []
    for resp in rec.get("responses", []):
        for pkt in resp.get("packets", []):
            try:
                data = json.loads(pkt.get("value", ""))
            except ValueError:
                continue
            delta = (data.get("choices") or [{}])[0].get("delta") or {}
            text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            usage = data.get("usage") or {}
            out.append((resp["perf_ns"], bool(text), usage))
    return out


rows, native, errors = [], [], 0
for line in Path(run_dir, "profile_export_raw.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if rec["metadata"].get("benchmark_phase") != "profiling":
        continue
    chunks = content_chunks(rec)
    content = [c for c in chunks if c[1]]
    usages = [c[2] for c in chunks if c[2]]
    ok = rec.get("status") == 200 and not rec.get("error") and content and usages
    if not ok:
        errors += 1
        native.append({"terminal_status": "failed", "output_length": 0})
        continue
    start = rec["start_perf_ns"]
    osl, isl = usages[-1]["completion_tokens"], usages[-1]["prompt_tokens"]
    first_chunk = content[0][2].get("completion_tokens") if per_chunk and content[0][2] else None
    native.append(
        {
            "terminal_status": "completed",
            "arrival_time_ms": 0.0,
            "output_length": osl,
            "first_token_ms": (content[0][0] - start) / 1e6,
            "last_token_ms": (content[-1][0] - start) / 1e6,
            "first_content_chunk_tokens": first_chunk,
        }
    )
    rows.append((rec["metadata"]["request_start_ns"], isl, osl))

agg = json.loads(Path(run_dir, "profile_export_aiperf.json").read_text())
ours = summarize_requests(native, per_chunk_usage=per_chunk)
calc = {}
for name in NAMES:
    ref = agg.get(name) or {}
    calc[f"{name}.count"] = (ours[f"aiperf.{name}.count"], float(ref.get("count", 0)))
    for s in STATS:
        calc[f"{name}.{s}"] = (ours.get(f"aiperf.{name}.{s}"), ref.get(s))
max_rel = max(abs(a - b) / max(abs(b), 1e-12) for a, b in calc.values() if a is not None and b is not None)

# Prediction: same observed lengths, dispatch order, same concurrency and usage mode.
os.makedirs(work, exist_ok=True)
trace = os.path.join(work, "observed_trace.jsonl")
nxt = 0
with open(trace, "w") as f:
    for _, isl, osl in sorted(rows):
        k = math.ceil(isl / 64)
        f.write(json.dumps({"input_length": isl, "output_length": osl, "hash_ids": list(range(nxt, nxt + k))}) + "\n")
        nxt += k
cfg = os.path.join(work, "predict.yaml")
with open(cfg, "w") as f:
    f.write(f"""traffic:
  source: {{type: trace, format: mooncake, paths: [{trace}], block_size: 64}}
  load: {{type: concurrency, concurrency: {conc}}}
engine:
  mode: aggregated
  model: {model}
  hardware: gb200
  backend: vllm
  backend_version: 0.30.0
  context_length: 4096
  workers:
    aggregated:
      parallelism: {{tensor: 1}}
      scheduler: {{max_batched_tokens: 8192, max_sequences: 64}}
      kv_cache: {{prefix_caching: false}}
      timing: {{type: default}}
evaluation:
  aiperf: {{per_chunk_usage: {str(per_chunk).lower()}}}
""")
proc = subprocess.run(
    [
        sys.executable,
        "-m",
        "aisimulate",
        "predict",
        "-c",
        cfg,
        "--format",
        "json",
        "--output-dir",
        os.path.join(work, "predict"),
        "--overwrite",
    ],
    capture_output=True,
    text=True,
)
if proc.returncode:
    sys.exit(f"predict failed: {proc.stderr[-2000:]}")
pred = json.loads(proc.stdout[proc.stdout.index("{") :])
err = {}
for name in NAMES:
    for s in STATS:
        ref, sim = agg.get(name, {}).get(s), pred.get(f"aiperf.{name}.{s}")
        err[f"{name}.{s}"] = (sim, ref, None if ref in (None, 0) or sim is None else (sim - ref) / ref)
result = {
    "run": os.path.basename(run_dir.rstrip("/")),
    "model": model,
    "revision": rev,
    "concurrency": conc,
    "per_chunk_usage": per_chunk,
    "records": len(native),
    "errors": errors,
    "aiperf_error_summary": agg.get("error_summary"),
    "aiperf_request_count": (agg.get("request_count") or {}).get("avg"),
    "calc_max_rel_diff": max_rel,
    "calculation": calc,
    "sim_counts": {n: pred.get(f"aiperf.{n}.count") for n in NAMES},
    "prediction": err,
}
Path(work, "compare.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: result[k] for k in ("run", "model", "records", "errors", "calc_max_rel_diff", "sim_counts")}))
for k, (sim, ref, rel) in err.items():
    print(f"  {k:42s} aiperf={ref!s:>22} sim={sim!s:>22} rel_err={'' if rel is None else f'{rel:+.3f}'}")
