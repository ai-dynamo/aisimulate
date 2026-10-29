# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""macOS stdio adapter for existing revision-local forward gate workers.

Adapted from the same repository at baff0871,
benchmarks/evidence/afd-unification/measure_forward.py.
No taskset is available on macOS. Worker protocol, priming, timing and gate
comparison remain the repository's implementations. Run sequentially after
builds/tests; CPU affinity is explicitly unavailable.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/aisimulate"))
from tools.forward_perf_gate import PROTOCOL_VERSION, cases, compare

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument(
    "--candidate-revision", help="immutable source snapshot verified by the caller"
)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
BASE = args.baseline.resolve()
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip()
head = (
    args.candidate_revision
    or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
)
selected = cases.expand_cases()
arms = {
    "base": (
        BASE / "python/aisimulate/tools/forward_perf_gate/worker.py",
        BASE / "python/aisimulate/src",
        sha,
    ),
    "head": (
        ROOT / "python/aisimulate/tools/forward_perf_gate/worker.py",
        ROOT / "python/aisimulate/src",
        head,
    ),
}
identities = {}
for arm, (_, source, revision) in arms.items():
    env = {**os.environ, "PYTHONPATH": str(source)}
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import hashlib,json,pathlib; from aisimulate import _runtime as r; "
            "p=pathlib.Path(r.__file__).resolve(); digest=hashlib.sha256(p.read_bytes()).hexdigest(); "
            "print(json.dumps(dict(runtime=str(p), sha256=digest)))",
        ],
        env=env,
        text=True,
        cwd=source.parents[2],
    )
    identities[arm] = json.loads(probe)
    assert (
        pathlib.Path(identities[arm]["runtime"]).parent
        == (source / "aisimulate").resolve()
    )
raw = {
    "schema_version": 1,
    "native_identities": identities,
    "base_revision": sha,
    "head_revision": head,
    "adapter_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    "host": {"platform": platform.platform()},
    "configuration": {
        "mode": "local-macos-full-matrix",
        "rounds": 5,
        "warmup": 10,
        "iterations": 100,
        "cpu_affinity": "unavailable on macOS",
        "matrix_case_count": len(selected),
        "selected_case_count": len(selected),
        "expected_case_ids": [c["case_id"] for c in selected],
    },
    "run_errors": [],
    "cases": [{"case": c, "rounds": []} for c in selected],
}
for round_index in range(-1, 5):
    responses = {}
    for arm in ("base", "head") if round_index % 2 == 0 else ("head", "base"):
        worker, source, revision = arms[arm]
        env = {
            **os.environ,
            "PYTHONPATH": str(source),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "AIC_ALLOW_UNLISTED_VERSIONS": "1",
        }
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "revision": revision,
            "cases": selected if round_index % 2 == 0 else selected[::-1],
            "warmup": 0 if round_index < 0 else 10,
            "iterations": 1 if round_index < 0 else 100,
        }
        p = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        results = json.loads(p.stdout)["results"]
        responses[arm] = {r["case_id"]: r for r in results}
    if round_index < 0:
        assert all(len(rows) == len(selected) for rows in responses.values())
        assert all(
            row.get("status") == "OK"
            for rows in responses.values()
            for row in rows.values()
        ), responses
        raw["prewarm"] = responses
        print("availability pass complete", flush=True)
        continue
    for entry in raw["cases"]:
        cid = entry["case"]["case_id"]
        r = {"round": round_index, **{a: responses[a][cid] for a in arms}}
        assert all(r[a].get("status") == "OK" for a in arms), r
        assert math.isclose(
            r["base"]["predicted_value_ms"],
            r["head"]["predicted_value_ms"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ), r
        entry["rounds"].append(r)
    (OUT / "raw-results.json").write_text(json.dumps(raw, indent=2) + "\n")
    print("forward round", round_index + 1, flush=True)
comparison = compare.compare_raw(raw)
compare.write_outputs(comparison, OUT)
print(compare.render_markdown(comparison))
assert not comparison["blocking"], "forward gate requires investigation"

for filename in ("raw-results.json", "comparison.json"):
    path = OUT / filename
    path.with_suffix(path.suffix + ".gz").write_bytes(
        gzip.compress(path.read_bytes(), mtime=0)
    )
    path.unlink()
