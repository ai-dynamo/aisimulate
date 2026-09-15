# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
progress = {"worker": socket.gethostname(), "started_at": datetime.datetime.now(datetime.UTC).isoformat(), "shards": []}


def save():
    temporary = ROOT / "progress.tmp.json"
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(ROOT / "progress.json")


save()
for shard in json.loads((ROOT / "assignment.json").read_text()):
    system, backend = shard["system"], shard["backend"]
    output = ROOT / "results" / system / backend
    output.mkdir(parents=True)
    entry = {**shard, "status": "running", "started_at": datetime.datetime.now(datetime.UTC).isoformat()}
    progress["shards"].append(entry)
    save()
    started = time.monotonic()
    command = [
        sys.executable,
        str(ROOT / "run_shard.py"),
        "--system",
        system,
        "--backend",
        backend,
        "--max-workers",
        "8",
        "--output-dir",
        str(output),
    ]
    with (output / "execution.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=480 * 60)
            entry["returncode"] = result.returncode
            entry["status"] = "complete" if result.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            entry["status"] = "timed_out"
    entry["elapsed_seconds"] = time.monotonic() - started
    entry["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    save()
progress["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
save()
