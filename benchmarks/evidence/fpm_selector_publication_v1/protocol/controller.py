# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import time
from pathlib import Path

r = Path("/home/yueqiz/work/fpm-model-publication-v1")
py = "/home/yueqiz/work/fpm-beta-env/bin/python"
out = r / "cost-results"
out.mkdir(exist_ok=True)
for arm in ("base", "online", "online-before"):
    p = r / (arm + "-python")
    p.write_text(
        "#!/bin/sh\nexport PYTHONPATH="
        + str(r / arm / "python/aisimulate/src")
        + "\nexec "
        + py
        + ' "$@"\n'
    )
    p.chmod(0o755)
env = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def run(cmd, path):
    print(time.strftime("%H:%M:%S"), path.name, flush=True)
    with path.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)


run(
    [
        "taskset",
        "-c",
        "2",
        str(r / "online-python"),
        str(r / "online/python/aisimulate/tools/forward_perf_gate/run.py"),
        "--base-python",
        str(r / "base-python"),
        "--base-worker",
        str(r / "base/python/aisimulate/tools/forward_perf_gate/worker.py"),
        "--base-revision",
        "56f47543a15ae1f7842ecc5bcf653409c138d496",
        "--head-python",
        str(r / "online-python"),
        "--head-worker",
        str(r / "online/python/aisimulate/tools/forward_perf_gate/worker.py"),
        "--head-revision",
        "470898a39f526aaa0af479ba89092a92b5883ea8",
        "--output-dir",
        str(r / "forward-gate"),
        "--smoke",
        "--rounds",
        "5",
        "--warmup",
        "10",
        "--iterations",
        "100",
    ],
    r / "forward-gate.log",
)
for i in range(3):
    order = [
        ("base", "base"),
        ("online", "frozen"),
        ("online-before", "online"),
        ("online", "online"),
    ]
    if i % 2:
        order.reverse()
    for source, arm in order:
        run(
            [
                "taskset",
                "-c",
                "2",
                str(r / (source + "-python")),
                str(r / "benchmark.py"),
                arm,
                str(r / "fixture"),
            ],
            out / f"{source}-{arm}-{i}.json",
        )
run(
    [
        "taskset",
        "-c",
        "2",
        str(r / "online-python"),
        str(r / "benchmark.py"),
        "prepare",
        str(r / "small-fixture"),
        "--history-rows",
        "256",
    ],
    r / "small-fixture.json",
)
for i in range(3):
    order = ["online-before", "online"] if i % 2 == 0 else ["online", "online-before"]
    for source in order:
        run(
            [
                "taskset",
                "-c",
                "2",
                str(r / (source + "-python")),
                str(r / "benchmark.py"),
                "online",
                str(r / "small-fixture"),
            ],
            out / f"small-{source}-{i}.json",
        )
print("complete", flush=True)
