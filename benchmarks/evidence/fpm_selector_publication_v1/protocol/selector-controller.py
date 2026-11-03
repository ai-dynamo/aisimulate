# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

r = Path("/home/yueqiz/work/fpm-model-publication-v1")
py = "/home/yueqiz/work/fpm-beta-env/bin/python"
p = r / "selector-python"
p.write_text(
    "#!/bin/sh\nexport PYTHONPATH="
    + str(r / "selector/python/aisimulate/src")
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
out = r / "selector-cost-results"
out.mkdir(exist_ok=True)


def run(cmd, path):
    print(time.strftime("%H:%M:%S"), path.name, flush=True)
    with path.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)


run(
    ["taskset", "-c", "2", str(p), str(r / "cross-version.py"), "read"],
    r / "cross-version-child.json",
)
a = json.loads((r / "cross-version-parent.json").read_text())
b = json.loads((r / "cross-version-child.json").read_text())
assert a == b
(r / "cross-version-result.json").write_text(
    json.dumps(
        {
            "equal": True,
            "accepted": a["accepted"],
            "prediction_count": len(a["predictions"]),
            "parent_output_sha256": hashlib.sha256(
                (r / "cross-version-parent.json").read_bytes()
            ).hexdigest(),
            "child_output_sha256": hashlib.sha256(
                (r / "cross-version-child.json").read_bytes()
            ).hexdigest(),
        },
        indent=2,
    )
)
for fixture in ("fixture", "small-fixture"):
    for i in range(3):
        order = (
            [("online", "online"), ("selector", "selector")]
            if i % 2 == 0
            else [("selector", "selector"), ("online", "online")]
        )
        for source, arm in order:
            run(
                [
                    "taskset",
                    "-c",
                    "2",
                    str(r / (source + "-python")),
                    str(r / "selector-benchmark.py"),
                    arm,
                    str(r / fixture),
                ],
                out / f"{fixture}-{arm}-{i}.json",
            )
print("complete", flush=True)
